/**
 * Application orchestrator.
 *
 * Owns the pipeline:
 *
 *   camera -> frame scheduler -> preprocessor -> detector (worker)
 *          -> tracker -> free space -> corridor -> risk -> path scores
 *          -> decision -> speech
 *
 * with the health monitor, watchdog, recovery manager and degradation controller
 * running alongside.
 *
 * Two structural rules are enforced here and are worth stating explicitly:
 *
 *  1. **The render loop never awaits inference.** `requestAnimationFrame` is
 *     re-scheduled at the top of every callback and inference is fired without
 *     being awaited. The camera preview and the UI therefore cannot be blocked by
 *     the model, which was the single biggest cause of the freezing this upgrade
 *     set out to fix.
 *
 *  2. **There is no frame queue.** The scheduler refuses to submit while a
 *     previous inference is in flight, and refuses again if no tensor buffer is
 *     free. Skipped frames are counted, never buffered.
 */

import { QualityMode, buildDeviceProfile, resolveProfile } from "./config/deviceProfiles.js";
import { VisionConfig } from "./config/visionConfig.js";
import { SpeechConfig, SpeechPriority } from "./config/speechConfig.js";

import { CameraManager } from "./camera/cameraManager.js";
import { CoordinateMapper, ObjectFit } from "./vision/coordinateMapper.js";
import { FramePipeline } from "./vision/framePipeline.js";
import { FrameScheduler, DropReason } from "./vision/frameScheduler.js";
import { FrameQualityMonitor } from "./vision/frameQuality.js";
import { SceneChangeDetector } from "./vision/sceneChange.js";
import { ObjectTracker } from "./vision/tracker.js";
import { InferenceClient, SkipReason } from "./vision/inferenceClient.js";

import { FreeSpaceEstimator } from "./perception/freeSpace.js";
import { AdaptiveSegmentationController, GroundFreenessEstimator } from "./perception/segmentationStage.js";

import { CorridorModel } from "./navigation/corridorModel.js";
import { RiskEngine } from "./navigation/riskEngine.js";
import { PathScoreEngine } from "./navigation/pathScoring.js";
import { Action, DecisionEngine } from "./navigation/decisionEngine.js";

import { SpeechManager } from "./audio/speechManager.js";

import { VisionHealthMonitor } from "./health/visionHealthMonitor.js";
import { RecoveryManager, RecoveryStep, VisionWatchdog } from "./health/recoveryManager.js";
import { DegradationController, DegradationLevel } from "./health/degradation.js";

import { StatusView, UiState } from "./ui/statusView.js";
import { AssistanceControls } from "./ui/assistanceControls.js";
import { DebugOverlay } from "./ui/debugOverlay.js";
import { PerformanceDashboard } from "./ui/performanceDashboard.js";
import { BenchmarkScreen } from "./ui/benchmarkScreen.js";
import { DemoModeController, StartupStage } from "./ui/demoMode.js";
import { ScenarioRunnerUI } from "./ui/scenarioRunner.js";

/** How often the luminance-derived analysis stages run, in inference cycles. */
const ANALYSIS_EVERY_N_INFERENCES = 1;

class NavigationApp {
    constructor() {
        this.video = document.getElementById("camera-video-feed");
        this.overlayCanvas = document.getElementById("debug-canvas-overlay");

        this.statusView = new StatusView();
        this.mapper = new CoordinateMapper();

        this.camera = new CameraManager(this.video, {
            onGeometryChange: (geometry) => this._applyGeometry(geometry),
            onTrackEnded: () => this._handleCameraLoss()
        });

        this.speech = new SpeechManager({
            onAnnounce: (text) => this.statusView.announceGuidance(text)
        });

        this.health = new VisionHealthMonitor();
        this.degradation = new DegradationController({
            onChange: (level, detail) => this._onDegradationChange(level, detail)
        });

        this.tracker = new ObjectTracker();
        this.corridor = new CorridorModel();
        this.freeSpace = new FreeSpaceEstimator();
        this.riskEngine = new RiskEngine();
        this.pathScores = new PathScoreEngine();
        this.decisions = new DecisionEngine();
        this.quality = new FrameQualityMonitor();
        this.sceneChange = new SceneChangeDetector();

        this.refinement = new AdaptiveSegmentationController({
            estimator: new GroundFreenessEstimator(),
            enabled: true
        });

        // Populated by _bootstrap once the device has been profiled.
        this.capabilities = null;
        this.profile = resolveProfile();
        this.pipeline = null;
        this.scheduler = null;
        this.inference = null;

        this.overlay = new DebugOverlay(this.overlayCanvas, { mapper: this.mapper });
        this.dashboard = new PerformanceDashboard(document.getElementById("diagnostics-panel"));
        this.demo = new DemoModeController({ container: document.getElementById("demo-course-panel") });

        this.benchmark = new BenchmarkScreen({
            container: document.getElementById("benchmark-panel"),
            video: this.video,
            getContext: () => ({ capabilities: this.capabilities, profile: this.profile }),
            onBeforeRun: () => this._suspendForBenchmark(),
            onAfterRun: () => this._resumeAfterBenchmark()
        });

        this.scenarioRunner = new ScenarioRunnerUI({
            select: document.getElementById("select-test-scenario"),
            runButton: document.getElementById("btn-run-scenario"),
            runAllButton: document.getElementById("btn-run-all-scenarios"),
            log: document.getElementById("scenario-output-log")
        });

        this.controls = new AssistanceControls({
            onStart: () => this.startAssistance(),
            onPause: () => this.pauseAssistance(),
            onResume: () => this.resumeAssistance(),
            onStop: () => this.stopAssistance(),
            onToggleMute: () => this.toggleMute(),
            onToggleDebug: () => this.toggleDebug(),
            onToggleDiagnostics: () => this.toggleDiagnostics(),
            onToggleBenchmark: () => this.toggleBenchmark(),
            onToggleDemo: () => this.toggleDemoMode(),
            onDescribeScene: () => this.describeScene(),
            onRetry: () => this.retryVision(),
            onQualityModeChange: (mode) => this.setQualityMode(mode)
        });

        this.watchdog = null;
        this.recovery = null;

        this.state = UiState.IDLE;
        this.isAssisting = false;
        this.isPaused = false;
        this.suspended = false;
        this.animationFrameId = null;
        this.inferenceCount = 0;
        this.lastDecision = null;
        this.lastFreeSpaceReport = null;
        this.lastPathScores = null;
        this.lastTracks = [];
        this.lastSceneResult = null;

        this._bootstrap();
    }

    /* ------------------------------------------------------------ bootstrap */

    async _bootstrap() {
        this._setState(UiState.INITIALIZING);
        this.controls.setStartEnabled(false, "Vision system is still starting");
        this.statusView.setProgress({ ratio: 0, label: "Profiling device…" });

        this._installLifecycleHandlers();

        try {
            const { capabilities, tier, profile } = await buildDeviceProfile();
            this.capabilities = capabilities;
            this.tier = tier;
            this.profile = profile;
            this.demo.markStage(StartupStage.DEVICE_PROFILE, true);
            this.refinement.enabled = profile.allowSegmentation;
            if (!profile.allowSegmentation) {
                this.refinement.disabledReason = `disabled for ${tier} device profile`;
            }

            this.pipeline = new FramePipeline({ inputSize: profile.inputSize });
            this.scheduler = new FrameScheduler(VisionConfig.scheduler, profile);
            this.inference = new InferenceClient({
                capabilities,
                profile,
                framePipeline: this.pipeline,
                onEvent: (name, detail) => this._onInferenceEvent(name, detail)
            });

            this._installRecovery();

            this.statusView.setProgress({ ratio: 0.15, label: "Preparing vision model…" });
            const ready = await this._prepareDetector();
            if (!ready) return;

            this.demo.markStage(StartupStage.WARMUP, true);
            this.demo.markStage(StartupStage.HEALTH_CHECK, true);

            this._setState(UiState.VISION_READY);
            this.statusView.hideProgress();
            this.controls.setStartEnabled(true);
            this.speech.speakSystem(SpeechConfig.phrases.SYSTEM_READY, SpeechPriority.SYSTEM);
        } catch (error) {
            console.error("[app] Startup failed:", error);
            this._enterErrorState(String(error?.message || error));
        }
    }

    async _prepareDetector() {
        this.health.markModelStatus("loading");
        const result = await this.inference.initialize();

        if (!result.ok) {
            const summary = (result.attempts || [])
                .map((a) => `${a.backendId}/${a.modelKey}: ${a.reason}`)
                .join(" · ");
            console.error("[app] No detector could be prepared.", result.attempts);
            this.degradation.degradeTo(DegradationLevel.UNRELIABLE, "no detector passed the warm-up gate");
            this._enterErrorState(summary || "no detector available");
            return false;
        }

        this.health.markModelStatus(
            result.initInfo?.fromCache ? "ready (cached)" : "ready (downloaded)",
            this.inference.backendId
        );
        this.health.markWorkerStatus(this.inference.adapter?.kind === "worker" ? "running" : "main-thread");
        this.demo.markStage(StartupStage.MODEL, true);

        // The warm-up gate has already run inside initialize(); record the result.
        const warm = result.warmup;
        console.info(
            `[app] Detector ready: ${this.inference.model.label} on ${this.inference.backendLabel}`,
            `first pass ${warm?.firstPassMs} ms, steady state ${warm?.steadyStateMs} ms`,
            result.initInfo?.fromCache ? "(from cache)" : "(downloaded)"
        );
        return true;
    }

    _installRecovery() {
        this.watchdog = new VisionWatchdog({
            monitor: this.health,
            onStall: (detail) => this._handleStall(detail)
        });

        this.recovery = new RecoveryManager({
            monitor: this.health,
            onProgress: ({ step, attempt }) => {
                this._setState(UiState.RECOVERING, `${step} (${attempt})`);
                if (attempt === 1) this.speech.speakSystem(SpeechConfig.phrases.RECOVERING, SpeechPriority.SYSTEM);
            },
            onExhausted: () => this._enterErrorState("recovery exhausted"),
            handlers: {
                // Cheapest first: the pipeline may simply have hiccuped.
                [RecoveryStep.RETRY]: async () => ({ ok: this.inference?.ready === true }),
                [RecoveryStep.RESTART_WORKER]: async () => {
                    this.health.markWorkerStatus("restarting");
                    const result = await this.inference.restart();
                    this.health.markWorkerStatus(result.ok ? "running" : "failed");
                    return { ok: result.ok, detail: result.attempt?.reason };
                },
                [RecoveryStep.FALLBACK_BACKEND]: async () => {
                    const result = await this.inference.fallbackToNextBackend();
                    if (result.ok) {
                        this.health.markModelStatus("ready (fallback)", this.inference.backendId);
                        this.degradation.degradeTo(
                            DegradationLevel.NO_REFINEMENT,
                            `fell back to ${this.inference.backendLabel}`
                        );
                    }
                    return { ok: result.ok, detail: result.reason || result.attempt?.reason };
                },
                [RecoveryStep.RELOAD_MODEL]: async () => {
                    // Drop the cached bytes in case they are the problem, then
                    // rebuild from scratch.
                    const { clearModelCache } = await import("./vision/modelCache.js");
                    await clearModelCache();
                    const result = await this.inference.restart();
                    return { ok: result.ok, detail: "cache cleared" };
                },
                [RecoveryStep.REINITIALISE]: async () => {
                    this.tracker.clear();
                    this.decisions.reset();
                    const result = await this.inference.initialize();
                    return { ok: result.ok };
                }
            }
        });
    }

    /* ------------------------------------------------------------ lifecycle */

    async startAssistance() {
        if (this.isAssisting) return;
        if (this.state === UiState.VISION_ERROR) {
            this.retryVision();
            return;
        }

        this._setState(UiState.INITIALIZING, "starting camera");
        this.statusView.setProgress({ ratio: 0.6, label: "Starting camera…" });

        try {
            await this.camera.start();
            this.demo.markStage(StartupStage.CAMERA, true);
            this._applyGeometry(this.camera.geometry);

            if (this.camera.facing === "user") {
                console.warn("[app] Using a user-facing camera; the scene is behind the user.");
                this.statusView.announceStatus("Warning: front camera in use. Guidance will not reflect the path ahead.");
            }
        } catch (error) {
            console.error("[app] Camera unavailable:", error);
            this.statusView.hideProgress();
            this._setState(UiState.VISION_ERROR, this.camera.lastError || "camera unavailable");
            this.speech.speakSystem(SpeechConfig.phrases.CAMERA_UNAVAILABLE, SpeechPriority.IMMEDIATE_DANGER);
            this.controls.setErrorState(true);
            return;
        }

        this.tracker.clear();
        this.decisions.reset();
        this.quality.reset();
        this.sceneChange.reset();
        this.scheduler.reset();
        this.health.resetSession();
        this.recovery.reset();
        this.speech.reset();

        this.statusView.hideProgress();
        this.isAssisting = true;
        this.isPaused = false;
        this.controls.setAssisting(true);
        this._setState(UiState.ASSISTANCE_ACTIVE);
        this.speech.speakSystem(SpeechConfig.phrases.ASSISTANCE_STARTED, SpeechPriority.SYSTEM);

        this.watchdog.start();
        this._startLoop();
    }

    pauseAssistance() {
        if (!this.isAssisting || this.isPaused) return;
        this.isPaused = true;
        this.controls.setPaused(true);
        this.watchdog.stop();
        this._setState(UiState.PAUSED);
        this.speech.speakSystem(SpeechConfig.phrases.ASSISTANCE_PAUSED, SpeechPriority.SYSTEM);
    }

    resumeAssistance() {
        if (!this.isAssisting || !this.isPaused) return;
        this.isPaused = false;
        this.controls.setPaused(false);
        // Everything temporal is stale after a pause; start clean rather than
        // reasoning from pre-pause geometry.
        this.tracker.clear();
        this.decisions.reset();
        this.sceneChange.reset();
        this.health.resetSession();
        this.watchdog.start();
        this._setState(UiState.ASSISTANCE_ACTIVE);
        this.speech.speakSystem(SpeechConfig.phrases.ASSISTANCE_RESUMED, SpeechPriority.SYSTEM);
    }

    stopAssistance() {
        if (!this.isAssisting) return;
        this._stopLoop();
        this.watchdog.stop();
        this.camera.stop();
        this.tracker.clear();
        this.decisions.reset();
        this.speech.cancelAll();

        this.isAssisting = false;
        this.isPaused = false;
        this.controls.setAssisting(false);
        this.overlay.clear();
        this._setState(UiState.STOPPED);
        this.speech.speakSystem(SpeechConfig.phrases.ASSISTANCE_STOPPED, SpeechPriority.SYSTEM);
    }

    async retryVision() {
        this.controls.setErrorState(false);
        this.recovery?.reset();
        this.degradation.restoreTo(DegradationLevel.FULL, "user requested retry");
        this._setState(UiState.INITIALIZING, "retrying");
        this.statusView.setProgress({ ratio: 0.2, label: "Retrying vision system…" });

        const ok = await this._prepareDetector();
        if (ok) {
            this._setState(UiState.VISION_READY);
            this.statusView.hideProgress();
            this.controls.setStartEnabled(true);
            this.speech.speakSystem(SpeechConfig.phrases.SYSTEM_READY, SpeechPriority.SYSTEM);
        }
    }

    _installLifecycleHandlers() {
        document.addEventListener("visibilitychange", () => {
            if (document.hidden && this.isAssisting && !this.isPaused) {
                // A backgrounded tab gets throttled rAF and, on Chrome, paused
                // speech synthesis. Continuing would produce stale guidance.
                this.pauseAssistance();
                this.statusView.announceStatus("Paused because the app was backgrounded.");
            }
        });

        window.addEventListener("resize", () => this._applyGeometry(this.camera.geometry));
        window.addEventListener("pagehide", () => {
            this._stopLoop();
            this.camera.stop();
            this.speech.cancelAll();
        });
    }

    /* ----------------------------------------------------------------- loop */

    _startLoop() {
        this._stopLoop();

        const loop = (timestamp) => {
            // Re-schedule first: nothing below this line can prevent the next
            // frame from being considered.
            this.animationFrameId = requestAnimationFrame(loop);

            if (!this.isAssisting || this.suspended) return;

            this.scheduler.tickCamera(timestamp);

            if (this.isPaused || !this.camera.hasFrame) {
                this.scheduler.markDropped(DropReason.PAUSED);
                return;
            }

            const gate = this.scheduler.shouldRun(timestamp, {
                busy: this.inference.busy,
                paused: this.isPaused,
                canCapture: this.pipeline.canCapture
            });

            if (!gate.run) {
                this.scheduler.markDropped(gate.reason);
                return;
            }

            this.scheduler.markSubmitted(timestamp);
            // Fired, not awaited. This is the line that keeps the UI responsive.
            this._runInference(timestamp);
        };

        this.animationFrameId = requestAnimationFrame(loop);
    }

    _stopLoop() {
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    async _runInference(timestamp) {
        let result;
        try {
            result = await this.inference.detect(this.video, timestamp);
        } catch (error) {
            this._onInferenceFailure(String(error?.message || error));
            return;
        }

        if (!result || result.skipped) {
            if (result?.skipped && result.skipped !== SkipReason.NOT_READY) {
                this.scheduler.markDropped(result.skipped);
            }
            return;
        }
        if (result.error) {
            this._onInferenceFailure(result.error);
            return;
        }

        const completedAt = performance.now();
        this.scheduler.markCompleted(completedAt, result.latencyMs, result.totalMs || result.latencyMs);
        this.health.markInferenceSuccess();
        this.recovery.notifySuccess();
        this.inferenceCount += 1;

        try {
            this._reason(result, completedAt);
        } catch (error) {
            // A reasoning bug must not take the pipeline down with it.
            console.error("[app] Reasoning stage failed:", error);
            this.health.markInferenceFailure(`reasoning: ${error?.message || error}`);
        }

        this.scheduler.control(completedAt);
        this._evaluateDegradation();
        this.health.sampleMemory();
        this._updateDiagnostics(completedAt);
    }

    /* ------------------------------------------------------------ reasoning */

    /**
     * Everything between "we have detections" and "we may have spoken".
     * Synchronous and cheap: a few hundred microseconds for a couple of dozen
     * tracks.
     */
    _reason(result, nowMs) {
        /* --- frame analysis (quality, scene change, ground freeness) ------ */

        let sceneResult = this.lastSceneResult;
        let refinement = null;

        if (this.inferenceCount % ANALYSIS_EVERY_N_INFERENCES === 0) {
            const sample = this.pipeline.sampleLuminance(this.video);
            if (sample) {
                this.quality.update(sample.luma, sample.width, sample.height);
                sceneResult = this.sceneChange.update(sample.luma, sample.width, sample.height);
                this.lastSceneResult = sceneResult;

                refinement = this.refinement.maybeRun({
                    luma: sample.luma,
                    width: sample.width,
                    height: sample.height,
                    nowMs,
                    obstaclePresent: (this.lastDecision?.action || Action.CONTINUE) !== Action.CONTINUE,
                    // "Uncertain" means we committed to a side without much
                    // margin: exactly when extra floor information can change
                    // the outcome.
                    directionUncertain: Boolean(this.lastDecision?.isDirectional)
                        && (this.lastDecision.confidence ?? 1) < 0.7,
                    sceneChanged: Boolean(sceneResult?.event && sceneResult.event !== "STABLE"),
                    pipelineP95Ms: this.scheduler.metrics().latencyP95
                });
            }
        }

        const trustTracks = sceneResult ? sceneResult.trustTracks : true;

        /* --- camera space -> navigation space ---------------------------- */

        const navDetections = [];
        for (const detection of result.detections) {
            const mapped = this.mapper.cameraToNavigation(detection.box);
            if (mapped.fullyOutside) continue;
            navDetections.push({
                label: detection.label,
                classId: detection.classId,
                confidence: detection.confidence,
                box: mapped.box,
                visibility: mapped.visibility
            });
        }

        /* --- tracking ---------------------------------------------------- */

        const capabilities = this.degradation.capabilities;
        const tracked = this.tracker.update(navDetections, nowMs, { trustTracks });
        this.lastTracks = tracked.tracks;
        this.health.setTrackingCount(tracked.tracks.length);

        /* --- corridor, free space, risk, path scores --------------------- */

        this.corridor.update({
            freeSpace: this.lastFreeSpaceReport,
            tracks: tracked.tracks,
            aspect: this.mapper.visibleAspect,
            reset: !trustTracks
        });

        const freeSpaceReport = this.freeSpace.update(
            tracked.tracks,
            this.corridor.bounds,
            capabilities.freeSpaceRefinement ? refinement : null
        );
        this.lastFreeSpaceReport = freeSpaceReport;

        const risk = this.riskEngine.scoreAll(tracked.tracks, this.corridor);
        const pathScores = this.pathScores.evaluate(tracked.tracks, freeSpaceReport, this.corridor);
        this.lastPathScores = pathScores;

        /* --- decision ---------------------------------------------------- */

        // At degradation level 3 tracking is untrustworthy, so directional
        // guidance is withheld regardless of what the scores say.
        const allowDirection = capabilities.directionalGuidance
            && this.quality.allowsDirectionalGuidance;

        const decision = this.decisions.evaluate({
            tracks: tracked.tracks,
            threats: risk.threats,
            emergent: capabilities.approachDetection ? tracked.emergent : [],
            pathScores,
            freeSpace: freeSpaceReport,
            corridor: this.corridor,
            nowMs,
            qualityScore: this.quality.rollingScore,
            qualityAllowsDirection: allowDirection,
            qualityRequiresStop: this.quality.requiresStop
        });
        this.lastDecision = decision;

        /* --- speech and UI ---------------------------------------------- */

        const announceClear = this.decisions.shouldAnnounceClear(nowMs);
        const spoken = this.speech.speakDecision(decision, { announceClear });
        if (spoken.spoken && announceClear) this.decisions.markClearAnnounced();

        this._reflectState(decision);
        this.statusView.updateDecision(decision, {
            pathScores,
            degradationLabel: this.degradation.label,
            degradationLevel: this.degradation.level
        });

        this.overlay.render({
            tracks: tracked.tracks,
            corridor: this.corridor,
            freeSpace: freeSpaceReport,
            decision,
            pathScores,
            nowMs
        });
    }

    /**
     * Keep the status pill honest about low visibility without letting it fight
     * the decision card for attention.
     */
    _reflectState(decision) {
        if (!this.isAssisting || this.isPaused) return;
        if (this.state === UiState.RECOVERING) return;

        const target = this.quality.requiresStop || !this.quality.allowsDirectionalGuidance
            ? UiState.LOW_VISIBILITY
            : UiState.ASSISTANCE_ACTIVE;

        if (target !== this.state) {
            const detail = target === UiState.LOW_VISIBILITY
                ? (this.quality.dominantIssue || "poor frames")
                : null;
            this._setState(target, detail);
        }
        void decision;
    }

    /* ------------------------------------------------------------- health */

    _onInferenceFailure(message) {
        console.warn("[app] Inference failed:", message);
        this.scheduler.markFailed();
        this.health.markInferenceFailure(message);

        if (this.health.consecutiveInferenceFailures >= VisionConfig.health.maxConsecutiveFailures) {
            this._handleStall({ reason: "consecutive-failures", message });
        }
    }

    async _handleStall(detail) {
        if (!this.isAssisting && this.state !== UiState.VISION_READY) return;
        if (this.recovery.inProgress) return;

        console.warn("[app] Vision stall detected:", detail);
        this.watchdog.suppressFor(VisionConfig.health.watchdogTimeoutMs * 2);

        const result = await this.recovery.escalate(detail);
        if (result.exhausted && !result.ok) {
            this._enterErrorState("vision system unavailable");
            return;
        }
        if (result.ok && this.isAssisting) {
            this._setState(this.isPaused ? UiState.PAUSED : UiState.ASSISTANCE_ACTIVE);
        }
    }

    _handleCameraLoss() {
        if (!this.isAssisting) return;
        console.warn("[app] Camera track ended unexpectedly.");
        this.health.markInferenceFailure("camera track ended");
        this._setState(UiState.RECOVERING, "camera lost");
        this.camera.start().then(
            () => {
                this._applyGeometry(this.camera.geometry);
                this._setState(UiState.ASSISTANCE_ACTIVE);
            },
            () => {
                this._enterErrorState("camera lost");
                this.speech.speakSystem(SpeechConfig.phrases.CAMERA_UNAVAILABLE, SpeechPriority.IMMEDIATE_DANGER);
            }
        );
    }

    _evaluateDegradation() {
        const metrics = this.scheduler.metrics();
        let stability = null;
        if (this.lastTracks.length > 0) {
            let sum = 0;
            for (const track of this.lastTracks) sum += track.stability;
            stability = sum / this.lastTracks.length;
        }

        const evaluation = DegradationController.evaluate({
            detectorWorking: this.inference.ready && this.health.totalInferences > 0,
            refinementEnabled: this.refinement.enabled,
            inferenceFps: metrics.inferenceFps,
            trackingStability: stability,
            perceptionUsable: !this.quality.requiresStop
        });

        this.degradation.apply(evaluation);
        this.health.setDegradationLevel(this.degradation.level);

        // If we are shedding load and the optional stage is still on, turn it off
        // before reducing the inference rate any further.
        if (metrics.thermalThrottled && this.refinement.enabled) {
            this.refinement.disable("shedding load under sustained thermal pressure");
        }
    }

    _onDegradationChange(level, detail) {
        console.warn(`[app] Degradation level ${level}: ${detail.reason}`);

        if (level === DegradationLevel.UNRELIABLE) {
            this._enterErrorState(detail.reason);
            return;
        }
        if (level >= DegradationLevel.DETECTOR_ONLY && detail.from < DegradationLevel.DETECTOR_ONLY) {
            this.speech.speakSystem(SpeechConfig.phrases.DEGRADED, SpeechPriority.CAUTION);
        }
    }

    _enterErrorState(reason) {
        this._stopLoop();
        this.watchdog?.stop();
        this.isAssisting = false;
        this.controls.setAssisting(false);
        this.controls.setErrorState(true);
        this.controls.setStartEnabled(false, reason);
        this.statusView.hideProgress();
        this._setState(UiState.VISION_ERROR, reason);
        this.health.markDead(reason);
        // The one message the user must not miss.
        this.speech.speakSystem(SpeechConfig.phrases.VISION_UNAVAILABLE, SpeechPriority.STOP, { interrupt: true });
    }

    /* ---------------------------------------------------------------- UI */

    _setState(state, detail = null) {
        this.state = state;
        this.statusView.setState(state, detail);
    }

    _applyGeometry(geometry) {
        if (!geometry) return;
        this.mapper.setVideoGeometry(geometry.videoWidth, geometry.videoHeight);
        this.mapper.setDisplayGeometry(geometry.displayWidth, geometry.displayHeight, ObjectFit.COVER);
        this.mapper.setOrientation({
            rotationDegrees: geometry.rotationDegrees,
            mirrored: geometry.mirrored,
            facingUser: geometry.facingUser
        });
        this.video.classList.toggle("mirrored", Boolean(geometry.mirrored));
    }

    _onInferenceEvent(name, detail) {
        switch (name) {
            case "model-progress":
                if (detail.stage === "download" && detail.total) {
                    this.statusView.setProgress({
                        ratio: 0.15 + 0.5 * (detail.ratio || 0),
                        label: `Downloading model… ${Math.round((detail.ratio || 0) * 100)}%`
                    });
                } else if (detail.stage === "cache-hit") {
                    this.statusView.setProgress({ ratio: 0.6, label: "Model loaded from cache" });
                }
                break;
            case "selection-attempt":
                this.statusView.setProgress({
                    ratio: null,
                    label: `${detail.stage === "warmup" ? "Warming up" : "Loading"} ${detail.modelLabel}…`
                });
                break;
            case "selection-result":
                if (!detail.accepted) {
                    console.info(`[app] Rejected ${detail.modelKey} on ${detail.backendId}: ${detail.reason}`);
                }
                break;
            case "worker-crash":
                this.health.markWorkerStatus("crashed");
                this._handleStall({ reason: "worker-crash", ...detail });
                break;
            case "worker-log":
                if (detail.level === "warn" || detail.level === "error") {
                    console.warn(`[worker] ${detail.message}`, detail.detail || "");
                }
                break;
            default:
                break;
        }
    }

    _updateDiagnostics(nowMs) {
        if (!this.dashboard.visible) return;
        this.dashboard.update({
            nowMs,
            detector: this.inference.describe(),
            scheduler: this.scheduler.metrics(),
            tracking: this.tracker.metrics(),
            freeSpace: this.freeSpace.metrics(),
            pathScores: this.pathScores.metrics(),
            decision: this.decisions.metrics(),
            health: this.health.snapshot(),
            speech: this.speech.metrics(),
            recovery: this.recovery.metrics(),
            refinement: this.refinement.metrics(),
            quality: this.quality.metrics(),
            corridor: this.corridor.metrics(),
            scene: this.sceneChange.metrics(),
            degradation: this.degradation.metrics(),
            pipeline: this.pipeline.stats(),
            coords: this.mapper.describe(),
            maxRisk: this.lastTracks.reduce((m, t) => Math.max(m, t.risk || 0), 0)
        });
    }

    /* ----------------------------------------------------------- toggles */

    toggleMute() {
        this.speech.setEnabled(!this.speech.enabled);
        this.controls.setMuted(!this.speech.enabled);
    }

    toggleDebug() {
        const visible = this.overlay.toggle();
        this.controls.setToggleState("debug", visible);
    }

    toggleDiagnostics() {
        const visible = this.dashboard.setVisible(!this.dashboard.visible);
        this.controls.setToggleState("diagnostics", visible);
    }

    toggleBenchmark() {
        const visible = this.benchmark.setVisible(!this.benchmark.visible);
        this.controls.setToggleState("benchmark", visible);
    }

    toggleDemoMode() {
        const enabled = this.demo.setEnabled(!this.demo.enabled);
        this.demo.setVisible(enabled);
        this.controls.setToggleState("demo", enabled);
        if (!enabled) return;

        // Demo mode wants speech on and the overlay off, so the audience sees the
        // assistance behaviour rather than the developer view.
        this.speech.setEnabled(true);
        this.controls.setMuted(false);
        this.overlay.setVisible(false);
        this.controls.setToggleState("debug", false);

        // Requirement 44: start the camera automatically once the vision system
        // has passed its gate, so the operator only has to press one thing. The
        // model has already downloaded and warmed up by this point, which is what
        // stops the first ten seconds of a demonstration looking broken.
        if (this.state === UiState.VISION_READY && !this.isAssisting) {
            this.statusView.announceStatus("Demo mode: starting assistance.");
            this.startAssistance();
            return;
        }

        this.statusView.announceStatus(
            this.demo.gateSatisfied
                ? "Demo mode ready."
                : `Demo mode: waiting for ${this.demo.pendingStages.join(", ")}`
        );
    }

    async setQualityMode(mode) {
        if (!Object.values(QualityMode).includes(mode)) return;
        this.profile = resolveProfile(this.tier, mode);
        this.scheduler.minFps = this.profile.minFps;
        this.scheduler.ceilingFps = this.profile.maxFps;
        this.scheduler.targetFps = Math.min(this.profile.targetFps, this.profile.maxFps);
        this.scheduler.intervalMs = 1000 / this.scheduler.targetFps;
        this.refinement.enabled = this.profile.allowSegmentation && this.refinement.enabled;
        await this.inference.setDetectionConfig({
            maxDetections: Math.min(VisionConfig.detection.maxDetections, this.profile.maxDetections)
        });
        this.statusView.announceStatus(`Quality mode: ${mode.toLowerCase()}.`);
    }

    /**
     * Describe the scene on demand. This is the only place long descriptive
     * speech is produced, and only because the user explicitly asked.
     */
    describeScene() {
        const tracks = this.lastTracks.filter((t) => t.confirmed && t.risk >= 15);
        if (tracks.length === 0) {
            this.speech.speakDescription("Nothing detected nearby.");
            return;
        }

        const bySector = { left: [], centre: [], right: [] };
        const bounds = this.corridor.bounds;
        for (const track of tracks.slice(0, 6)) {
            const key = track.centerX < bounds.left ? "left" : (track.centerX > bounds.right ? "right" : "centre");
            bySector[key].push(`${track.label} ${String(track.distanceCategory).toLowerCase().replace("_", " ")}`);
        }

        const parts = [];
        for (const [sector, items] of Object.entries(bySector)) {
            if (items.length) parts.push(`${sector}: ${items.join(", ")}`);
        }
        this.speech.speakDescription(parts.join(". ") || "Nothing detected nearby.");
    }

    /* --------------------------------------------------------- benchmark */

    async _suspendForBenchmark() {
        this.suspended = true;
        this._wasAssisting = this.isAssisting;
        if (this.isAssisting) {
            this._stopLoop();
            this.watchdog.stop();
        }
        // The benchmark builds its own detector instances; release ours so they
        // are not competing for the same GPU and CPU.
        this.inference.dispose();
        this.speech.cancelAll();

        if (!this.camera.isActive) {
            try {
                await this.camera.start();
                this._applyGeometry(this.camera.geometry);
            } catch (error) {
                console.error("[app] Benchmark needs the camera:", error);
            }
        }
        this._setState(UiState.BENCHMARKING);
    }

    async _resumeAfterBenchmark() {
        this.suspended = false;
        const ok = await this._prepareDetector();
        if (!ok) return;

        if (this._wasAssisting) {
            this.watchdog.start();
            this._startLoop();
            this._setState(UiState.ASSISTANCE_ACTIVE);
        } else {
            this._setState(UiState.VISION_READY);
        }
    }
}

// Guarded so the module can be imported outside a browser. That is not a
// theoretical nicety: it lets the integrity test load the whole dependency graph
// and catch a mistyped export name, which is otherwise only discovered by opening
// the page on a phone.
if (typeof window !== "undefined" && typeof document !== "undefined") {
    window.addEventListener("DOMContentLoaded", () => {
        window.navigationApp = new NavigationApp();
    });
}

export { NavigationApp };
