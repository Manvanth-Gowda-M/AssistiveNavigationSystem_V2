/**
 * Real-Time AI Vision Navigation Assistant — Main Client Orchestrator
 * Connects edge perception, temporal tracking, corridor spatial analysis,
 * risk scoring, directional decision engine, and prioritized speech synthesis.
 */

import { VisionConfig } from "./config/visionConfig.js";
import { NavigationConfig } from "./config/navigationConfig.js";
import { SpeechConfig } from "./config/speechConfig.js";

import { PerformanceTracker } from "./utils/timing.js";
import { CameraManager } from "./camera/cameraManager.js";
import { VisionDetector } from "./vision/visionDetector.js";
import { TemporalTracker } from "./vision/temporalTracker.js";

import { WalkingCorridor } from "./navigation/walkingCorridor.js";
import { SpatialAnalysis } from "./navigation/spatialAnalysis.js";
import { RiskEngine } from "./navigation/riskEngine.js";
import { NavigationDecisionEngine } from "./navigation/decisionEngine.js";
import { NavigationStateMachine, NavState } from "./navigation/stateMachine.js";

import { GuidanceSpeechEngine } from "./audio/speechEngine.js";
import { AudioFirstUIManager } from "./ui/audioFirstUI.js";
import { AssistanceControls } from "./ui/assistanceControls.js";
import { DebugOverlay } from "./ui/debugOverlay.js";
import { TestScenarios } from "./ui/scenarioRunner.js";

class NavigationApp {
    constructor() {
        this.videoElement = document.getElementById("camera-video-feed");
        this.debugCanvas = document.getElementById("debug-canvas-overlay");

        // Subsystems
        this.perf = new PerformanceTracker();
        this.camera = new CameraManager(this.videoElement);
        this.detector = new VisionDetector();
        this.tracker = new TemporalTracker();
        this.corridor = new WalkingCorridor();
        this.decisionEngine = new NavigationDecisionEngine();
        this.speechEngine = new GuidanceSpeechEngine();
        this.ui = new AudioFirstUIManager();
        this.debugOverlay = new DebugOverlay(this.debugCanvas);

        this.stateMachine = new NavigationStateMachine((newState, oldState, payload) => {
            this.handleStateChange(newState, oldState, payload);
        });

        this.controls = new AssistanceControls({
            onStart: () => this.startAssistance(),
            onPause: () => this.pauseAssistance(),
            onResume: () => this.resumeAssistance(),
            onStop: () => this.stopAssistance(),
            onToggleMute: () => this.toggleMute(),
            onToggleDebug: () => this.toggleDebug()
        });

        // Loop control
        this.animationFrameId = null;
        this.lastInferenceTime = 0;
        this.isProcessing = false;
        this.isScenarioRunning = false;

        this.init();
    }

    async init() {
        this.stateMachine.transition(NavState.INITIALIZING);
        this.ui.setStatus("Initializing Vision System...", "warning");

        try {
            await this.detector.initialize();
            this.stateMachine.transition(NavState.READY);
            this.ui.setStatus("Vision Ready", "ready");
            this.speechEngine.speakSystemMessage(SpeechConfig.phrases.SYSTEM_READY);
        } catch (err) {
            console.error("[NavigationApp] Initialization error:", err);
            this.stateMachine.transition(NavState.ERROR);
            this.ui.setStatus("Vision Error", "danger");
            this.speechEngine.speakSystemMessage(SpeechConfig.phrases.VISION_UNAVAILABLE);
        }

        this.initScenarioUI();
    }

    async startAssistance() {
        if (this.stateMachine.isAssisting()) return;

        this.ui.setStatus("Starting Camera...", "warning");
        try {
            await this.camera.startCamera();
            this.tracker.clear();
            this.stateMachine.transition(NavState.ASSISTING);
            this.controls.setAssistanceActive(true);
            this.ui.setStatus("Assisting", "active");
            this.speechEngine.speakSystemMessage(SpeechConfig.phrases.ASSISTANCE_STARTED);

            this.startLoop();
        } catch (err) {
            console.error("[NavigationApp] Camera start failed:", err);
            this.stateMachine.transition(NavState.ERROR);
            this.ui.setStatus("Camera Unavailable", "danger");
            this.speechEngine.speakSystemMessage(SpeechConfig.phrases.CAMERA_UNAVAILABLE);
        }
    }

    pauseAssistance() {
        this.stateMachine.transition(NavState.PAUSED);
        this.ui.setStatus("Paused", "warning");
        this.speechEngine.speakSystemMessage(SpeechConfig.phrases.ASSISTANCE_PAUSED);
    }

    resumeAssistance() {
        this.stateMachine.transition(NavState.ASSISTING);
        this.ui.setStatus("Assisting", "active");
    }

    stopAssistance() {
        this.stopLoop();
        this.camera.stopCamera();
        this.tracker.clear();
        this.speechEngine.cancelAll();
        this.stateMachine.transition(NavState.STOPPED);
        this.controls.setAssistanceActive(false);
        this.ui.setStatus("Stopped", "ready");
        this.ui.updateAction({ action: "STOPPED", urgency: "LOW", reason: "user_stopped" });
        this.speechEngine.speakSystemMessage(SpeechConfig.phrases.ASSISTANCE_STOPPED);
    }

    toggleMute() {
        this.speechEngine.isEnabled = !this.speechEngine.isEnabled;
        const btn = document.getElementById("btn-mute-audio");
        if (btn) {
            btn.textContent = this.speechEngine.isEnabled ? "🔊 Mute" : "🔇 Unmute";
            btn.className = this.speechEngine.isEnabled ? "btn btn-secondary" : "btn btn-warning";
        }
        if (!this.speechEngine.isEnabled) {
            this.speechEngine.cancelAll();
        }
    }

    toggleDebug() {
        const isVis = this.debugOverlay.toggle();
        const btn = document.getElementById("btn-toggle-debug");
        if (btn) {
            btn.className = isVis ? "btn btn-primary" : "btn btn-secondary";
        }
    }

    startLoop() {
        if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

        const loop = async (timestamp) => {
            if (!this.stateMachine.isAssisting() && !this.isScenarioRunning) return;

            this.perf.tick();

            // Quality / low-light check
            const quality = this.camera.checkQuality();
            if (quality.isLowLight && this.stateMachine.getState() !== NavState.LOW_VISIBILITY) {
                this.stateMachine.transition(NavState.LOW_VISIBILITY);
                this.speechEngine.speakSystemMessage(SpeechConfig.phrases.LOW_VISIBILITY);
            }

            // Throttled Edge Inference (approx 15 FPS)
            const timeSinceInference = timestamp - this.lastInferenceTime;
            if (timeSinceInference >= VisionConfig.inferenceIntervalMs && !this.detector.isBusy) {
                this.lastInferenceTime = timestamp;
                const t0 = performance.now();

                // 1. Detect objects
                const rawDetections = await this.detector.detectFrame(this.videoElement, timestamp);
                const latency = Math.round(performance.now() - t0);
                this.perf.recordInference(latency);

                // 2. Multi-frame Tracking & Persistence
                const activeTracks = this.tracker.update(rawDetections, timestamp);

                // 3. Risk & Spatial Evaluation
                for (const track of activeTracks) {
                    const corridorEval = this.corridor.evaluateObstacle(track.boundingBox);
                    RiskEngine.computeObstacleRisk(track, corridorEval);
                }

                // 4. Free Space Flank Clearance
                const flankClearance = SpatialAnalysis.evaluateFlankClearance(activeTracks, this.corridor.getBounds());

                // 5. Navigation Decision
                const confirmedTracks = activeTracks.filter(t => t.isConfirmed);
                const decision = this.decisionEngine.evaluate(confirmedTracks, this.corridor, flankClearance);

                // 6. Action-Oriented Speech Guidance
                this.speechEngine.speakDecision(decision);

                // 7. UI Updates
                this.ui.updateAction(decision);

                // 8. Debug Visualization
                if (this.debugOverlay.isVisible) {
                    this.debugOverlay.render({
                        videoWidth: this.videoElement.videoWidth || 640,
                        videoHeight: this.videoElement.videoHeight || 480,
                        corridor: this.corridor,
                        trackedObjects: activeTracks,
                        decision,
                        flankClearance,
                        fps: this.perf.getFps(),
                        latencyMs: this.perf.getAvgLatency(),
                        backend: this.detector.activeBackend
                    });
                }
            }

            this.animationFrameId = requestAnimationFrame(loop);
        };

        this.animationFrameId = requestAnimationFrame(loop);
    }

    stopLoop() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
    }

    handleStateChange(newState) {
        if (newState === NavState.EMERGENCY_STOP) {
            this.ui.setStatus("EMERGENCY STOP", "danger");
        } else if (newState === NavState.LOW_VISIBILITY) {
            this.ui.setStatus("Low Visibility", "warning");
        }
    }

    /**
     * Scenario Runner Setup
     */
    initScenarioUI() {
        const select = document.getElementById("select-test-scenario");
        const btnRun = document.getElementById("btn-run-scenario");
        const outputLog = document.getElementById("scenario-output-log");

        if (!select || !btnRun) return;

        select.innerHTML = TestScenarios.map(s => `<option value="${s.id}">${s.name}</option>`).join("");

        btnRun.addEventListener("click", async () => {
            const scenario = TestScenarios.find(s => s.id === select.value);
            if (!scenario) return;

            this.isScenarioRunning = true;
            this.tracker.clear();
            this.ui.setStatus(`Running: ${scenario.name}`, "active");
            if (outputLog) outputLog.innerHTML = `<div class="log-entry">▶ Running ${scenario.name}...</div>`;

            let simTime = Date.now();
            for (let f = 0; f < scenario.frames.length; f++) {
                simTime += 200;
                const simulatedDets = scenario.frames[f];
                this.detector.injectSimulatedDetections(simulatedDets);

                // Track & evaluate
                const activeTracks = this.tracker.update(this.detector.getDetections(), simTime);
                for (const t of activeTracks) {
                    RiskEngine.computeObstacleRisk(t, this.corridor.evaluateObstacle(t.boundingBox));
                }

                const flankClearance = SpatialAnalysis.evaluateFlankClearance(activeTracks, this.corridor.getBounds());
                const confirmed = activeTracks.filter(t => t.isConfirmed);
                const decision = this.decisionEngine.evaluate(confirmed, this.corridor, flankClearance);

                this.speechEngine.speakDecision(decision);
                this.ui.updateAction(decision);

                if (outputLog) {
                    outputLog.innerHTML += `<div class="log-entry">Frame ${f + 1}: Action=${decision.action} (${decision.urgency}) | Spoken="${this.speechEngine.generatePhrase(decision)}"</div>`;
                }

                await new Promise(r => setTimeout(r, 200));
            }

            if (outputLog) {
                outputLog.innerHTML += `<div class="log-entry log-success">✔ Scenario Complete. Expected: ${scenario.expectedAction}</div>`;
            }
            this.isScenarioRunning = false;
        });
    }
}

// Instantiate on DOM load
window.addEventListener("DOMContentLoaded", () => {
    window.app = new NavigationApp();
});
