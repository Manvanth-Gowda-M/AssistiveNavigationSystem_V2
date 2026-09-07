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
import { withTimeout } from "./utils/async.js";
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

/**
 * Hard ceiling on the whole startup sequence.
 *
 * Individual steps are bounded too, but this is the backstop that guarantees the
 * user is never left looking at "Initialising" with no explanation.
 */
const BOOTSTRAP_TIMEOUT_MS = 120000;

/**
 * Report a startup step to the on-page trace installed by `bootGuard.js`.
 *
 * Optional by design: the app must not depend on the diagnostic being present,
 * and the diagnostic must not depend on the app having loaded.
 */
function trace(name, status = "info", detail = "") {
    try {
        window.__vaBoot?.step(name, status, detail);
    } catch { /* diagnostics must never break startup */ }
    const line = detail ? `${name} — ${detail}` : name;
    if (status === "fail") console.error(`[startup] ${line}`);
    else if (status === "warn") console.warn(`[startup] ${line}`);
    else console.info(`[startup] ${line}`);
}

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
        trace("App constructed", "ok");

        this._installLifecycleHandlers();

        try {
            await withTimeout(this._bootstrapSteps(), BOOTSTRAP_TIMEOUT_MS, "startup");
        } catch (error) {
            const message = String(error?.message || error);
            trace("Startup aborted", "fail", message);
            window.__vaBoot?.fail(message);
            console.error("[app] Startup failed:", error);
            this._enterErrorState(message);
        }
    }

    async _bootstrapSteps() {
        /* ---------------------------------------------------- device profile */

        trace("Profiling device", "info");
        const { capabilities, tier, profile } = await buildDeviceProfile();
        this.capabilities = capabilities;
        this.tier = tier;
        this.profile = profile;
        this.demo.markStage(StartupStage.DEVICE_PROFILE, true);

        trace("Device profiled", "ok",
            `${tier} tier · ${profile.mode} · ${capabilities.hardwareConcurrency} cores`
            + ` · webgpu=${capabilities.hasWebGpu} · simd=${capabilities.hasWasmSimd}`
            + ` · threads=${capabilities.hasWasmThreads} · moduleWorker=${capabilities.hasWorker}`);

        if (capabilities.webGpuProbeNote) {
            trace("WebGPU probe", "warn", capabilities.webGpuProbeNote);
        }
        if (!capabilities.hasWorker) {
            trace("Module workers unavailable", "warn",
                "ONNX Runtime cannot be used; falling back to a main-thread detector.");
        }
        if (!capabilities.hasOffscreenCanvas2d) {
            trace("OffscreenCanvas 2D unavailable", "warn",
                "Using a detached canvas for frame preprocessing instead.");
        }

        this.refinement.enabled = profile.allowSegmentation;
        if (!profile.allowSegmentation) {
            this.refinement.disabledReason = `disabled for ${tier} device profile`;
        }

        /* ------------------------------------------------------- subsystems */

        this.pipeline = new FramePipeline({ inputSize: profile.inputSize });
        if (!this.pipeline.ready) {
            throw new Error("Could not create a 2D drawing context for frame preprocessing");
        }
        this.scheduler = new FrameScheduler(VisionConfig.scheduler, profile);
        this.inference = new InferenceClient({
            capabilities,
            profile,
            framePipeline: this.pipeline,
            onEvent: (name, detail) => this._onInferenceEvent(name, detail)
        });
        this._installRecovery();
        trace("Pipeline built", "ok",
            `input ${this.pipeline.inputSize}px · plan: ${this.inference.plan.map((p) => p.id).join(" → ")}`);

        /* --------------------------------------------------------- detector */

        this.statusView.setProgress({ ratio: 0.15, label: "Preparing vision model…" });
        const ready = await this._prepareDetector();
        if (!ready) return;

        this.demo.markStage(StartupStage.WARMUP, true);
        this.demo.markStage(StartupStage.HEALTH_CHECK, true);

        this._setState(UiState.VISION_READY);
        this.statusView.hideProgress();
        this.controls.setStartEnabled(true);
        this.speech.speakSystem(SpeechConfig.phrases.SYSTEM_READY, SpeechPriority.SYSTEM);

        window.__vaBoot?.done(
            `${this.inference.model.label} on ${this.inference.backendLabel}`
        );
    }

    async _prepareDetector() {
        this.health.markModelStatus("loading");
        const result = await this.inference.initialize();

        if (!result.ok) {
            // Every rejection reason is surfaced. Without this the user sees only
            // "Vision error" and has no way to tell a blocked CDN from a broken
            // GPU driver.
            for (const attempt of result.attempts || []) {
                trace(`Rejected ${attempt.modelLabel || attempt.modelKey}`, "fail",
                    `${attempt.backendLabel || attempt.backendId}: ${attempt.reason}`);
            }
            const summary = (result.attempts || [])
                .map((a) => `${a.backendId}/${a.modelKey}: ${a.reason}`)
                .join(" · ");
            console.error("[app] No detector could be prepared.", result.attempts);
            this.degradation.degradeTo(DegradationLevel.UNRELIABLE, "no detector passed the warm-up gate");
            window.__vaBoot?.fail(summary || "no detector available");
            this._enterErrorState(summary || "no detector available");
            return false;
        }

        this.health.markModelStatus(
            result.initInfo?.fromCache ? "ready (cached)" : "ready (downloaded)",
            this.inference.backendId
        );
        this.health.markWorkerStatus(this.inference.adapter?.kind === "worker" ? "running" : "main-thread");
        this.demo.markStage(StartupStage.MODEL, true);

        // Rejected candidates are still worth reporting: knowing that WebGPU was
        // tried and failed its warm-up gate is useful, not noise.
        for (const attempt of result.attempts || []) {
            if (attempt.accepted) continue;
            trace(`Rejected ${attempt.modelLabel || attempt.modelKey}`, "warn",
                `${attempt.backendLabel || attempt.backendId}: ${attempt.reason}`);
        }

        // The warm-up gate has already run inside initialize(); record the result.
        const warm = result.warmup;
        trace("Detector ready", "ok",
            `${this.inference.model.label} on ${this.inference.backendLabel}`
            + ` · ${result.initInfo?.fromCache ? "from cache" : "downloaded"}`
            + ` · warm-up first ${warm?.firstPassMs} ms, steady ${warm?.steadyStateMs} ms`);
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
            trace("Starting camera", "info");
            const result = await this.camera.start();
            this.demo.markStage(StartupStage.CAMERA, true);
            this._applyGeometry(this.camera.geometry);
            trace("Camera started", "ok",
                `${result.strategy || "reused"} · ${this.camera.describe().resolution} · facing ${this.camera.facing}`);

            if (this.camera.facing === "user") {
                trace("Front camera in use", "warn",
                    "The scene behind the user is not the path ahead; guidance will be wrong.");
                this.statusView.announceStatus("Warning: front camera in use. Guidance will not reflect the path ahead.");
            }
        } catch (error) {
            const message = String(error?.message || error);
            trace("Camera unavailable", "fail", this.camera.lastError || message);
            window.__vaBoot?.show();
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
            perceptionUsable: !this.quality.requiresStop,
            completedInferences: metrics.completed
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
        trace(`Degraded to level ${level}`, level >= DegradationLevel.DETECTOR_ONLY ? "warn" : "info", detail.reason);

        if (level === DegradationLevel.UNRELIABLE) {
            // Only a genuinely broken detector ends the session. Environmental
            // problems - darkness, motion blur, a featureless wall - are handled
            // at level 3 and recover on their own, so they must never land here.
            if (this.inference?.ready && this.health.totalInferences > 0) {
                console.warn("[app] Ignoring level 4 while the detector is still producing results.");
                this.degradation.restoreTo(DegradationLevel.DETECTOR_ONLY, "detector still alive");
                return;
            }
            this._enterErrorState(detail.reason);
            return;
        }
        if (level >= DegradationLevel.DETECTOR_ONLY && detail.from < DegradationLevel.DETECTOR_ONLY) {
            this.speech.speakSystem(SpeechConfig.phrases.DEGRADED, SpeechPriority.CAUTION);
        }
    }

    _enterErrorState(reason) {
        // Bring the startup trace back so the reason is visible, not just a
        // one-line status pill.
        try {
            window.__vaBoot?.show();
        } catch { /* diagnostics are optional */ }

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
                this._reportModelProgress(detail);
                break;
            case "selection-attempt":
                if (detail.stage === "init") {
                    trace(`Trying ${detail.modelLabel}`, "info", detail.backendLabel);
                }
                this.statusView.setProgress({
                    ratio: null,
                    label: `${detail.stage === "warmup" ? "Warming up" : "Loading"} ${detail.modelLabel}…`
                });
                break;
            case "selection-result":
                // Traced as it happens, not batched at the end. If selection stalls
                // or the overall budget runs out, the reasons gathered so far are
                // still on screen - which is the whole point of the trace.
                if (detail.accepted) {
                    trace(`Accepted ${detail.modelLabel}`, "ok",
                        `${detail.backendLabel} · steady p95 ${detail.p95} ms · first pass ${detail.firstPassMs} ms`);
                } else {
                    trace(`Rejected ${detail.modelLabel || detail.modelKey}`, "warn",
                        `${detail.backendLabel || detail.backendId}: ${detail.reason}`);
                }
                break;
            case "worker-crash":
                trace("Worker crashed", "fail", detail.message || detail.reason || "");
                this.health.markWorkerStatus("crashed");
                this._handleStall({ reason: "worker-crash", ...detail });
                break;
            case "worker-log":
                if (detail.level === "warn" || detail.level === "error") {
                    trace(detail.message, "warn", detail.detail || "");
                } else if (detail.message) {
                    trace(detail.message, "ok", detail.detail || "");
                }
                break;
            default:
                break;
        }
    }

    /**
     * Turn worker progress into both a progress bar and a trace entry.
     *
     * Download progress deliberately updates the bar only: a trace line per chunk
     * would bury everything else.
     */
    _reportModelProgress(detail) {
        switch (detail.stage) {
            case "worker-runtime":
                trace("Loading inference runtime", "info", "ONNX Runtime Web");
                this.statusView.setProgress({ ratio: null, label: "Loading inference runtime…" });
                break;
            case "worker-weights":
                this.statusView.setProgress({ ratio: 0.2, label: "Fetching model weights…" });
                break;
            case "worker-session":
                trace("Building inference session", "info");
                this.statusView.setProgress({ ratio: 0.7, label: "Building inference session…" });
                break;
            case "cache-hit":
                trace("Model loaded from cache", "ok");
                this.statusView.setProgress({ ratio: 0.6, label: "Model loaded from cache" });
                break;
            case "download":
                if (detail.total) {
                    const percent = Math.round((detail.ratio || 0) * 100);
                    this.statusView.setProgress({
                        ratio: 0.2 + 0.45 * (detail.ratio || 0),
                        label: `Downloading model… ${percent}% of ${(detail.total / 1e6).toFixed(1)} MB`
                    });
                } else {
                    this.statusView.setProgress({ ratio: null, label: "Downloading model…" });
                }
                break;
            case "persist":
                trace("Caching model", "ok", "will load from cache next time");
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
