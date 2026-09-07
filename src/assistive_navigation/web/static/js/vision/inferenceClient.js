/**
 * Inference client: backend/model selection, warm-up gating, and the single
 * in-flight submission discipline.
 *
 * This is the component that turns "we have several possible detectors" into
 * "we have one detector that has been proven to work on this device, right now".
 *
 * Selection algorithm:
 *   for each backend in the plan (WebGPU -> WASM SIMD -> WASM -> main-thread):
 *     for each model the backend can run, cheapest-first:
 *       create adapter, init, warm up, measure
 *       if the warm-up gate passes -> accept and stop
 *       otherwise dispose and continue
 *
 * Nothing is accepted on the basis of being *available*. Everything is accepted
 * on the basis of having been *measured* (requirements 5, 6).
 */

import {
    BackendId,
    BACKEND_LABELS,
    buildBackendPlan,
    evaluateWarmup
} from "./backendSelector.js";
import { Runtime, getModel, rankCandidates, resolveModelUrl } from "../config/modelRegistry.js";
import { VisionConfig } from "../config/visionConfig.js";
import { withTimeout } from "../utils/async.js";
import { WorkerDetectorAdapter } from "./adapters/workerAdapter.js";
import { MediaPipeDetectorAdapter } from "./adapters/mediapipeAdapter.js";
import { TfjsCocoSsdAdapter } from "./adapters/tfjsCocoSsdAdapter.js";

/** Reason codes surfaced to the health monitor and the dashboard. */
export const SkipReason = Object.freeze({
    NOT_READY: "not-ready",
    BUSY: "busy",
    NO_BUFFER: "no-buffer",
    NO_FRAME: "no-frame"
});

export class InferenceClient {
    /**
     * @param {{capabilities:object, profile:object, framePipeline:object,
     *          onEvent?:(name:string, detail:object)=>void}} opts
     */
    constructor({ capabilities, profile, framePipeline, onEvent = () => {} }) {
        this.capabilities = capabilities;
        this.profile = profile;
        this.framePipeline = framePipeline;
        this.onEvent = onEvent;

        this.plan = buildBackendPlan(capabilities);
        this.planIndex = 0;

        /** @type {object|null} */
        this.adapter = null;
        this.backendId = null;
        this.model = null;
        this.warmupResult = null;
        this.initInfo = null;
        this.ready = false;

        this.attempts = [];
        this.detectionConfig = {
            scoreThreshold: VisionConfig.detection.scoreThreshold,
            nmsIouThreshold: VisionConfig.detection.nmsIouThreshold,
            maxDetections: Math.min(VisionConfig.detection.maxDetections, profile.maxDetections),
            minBoxArea: VisionConfig.detection.minBoxArea
        };

        this._inFlight = false;
        this._lastLatencyMs = 0;
        this.submitted = 0;
        this.completed = 0;
        this.failed = 0;
        this.skippedBusy = 0;
    }

    get busy() {
        return this._inFlight;
    }

    get backendLabel() {
        return this.adapter?.backendLabel || BACKEND_LABELS[this.backendId] || "none";
    }

    /* ----------------------------------------------------------- adapters */

    _createAdapter(backendEntry, model) {
        const modelUrl = resolveModelUrl(model, document.baseURI);

        if (backendEntry.runtime === Runtime.ORT) {
            return new WorkerDetectorAdapter({
                model,
                modelUrl,
                backendId: backendEntry.id,
                capabilities: this.capabilities,
                detection: this.detectionConfig,
                inputSize: this.framePipeline.inputSize,
                onBufferReturn: (buf) => this.framePipeline.releaseBuffer(buf),
                onLog: (level, message, detail) => this.onEvent("worker-log", { level, message, detail }),
                onCrash: (detail) => this.onEvent("worker-crash", detail),
                onModelProgress: (p) => this.onEvent("model-progress", p)
            });
        }

        if (backendEntry.runtime === Runtime.MEDIAPIPE) {
            return new MediaPipeDetectorAdapter({
                model,
                modelUrl,
                delegate: backendEntry.id === BackendId.MEDIAPIPE_GPU ? "GPU" : "CPU",
                detection: this.detectionConfig
            });
        }

        return new TfjsCocoSsdAdapter({
            model,
            detection: this.detectionConfig,
            preferWebgl: backendEntry.id === BackendId.TFJS_WEBGL
        });
    }

    /**
     * Try one (backend, model) pair end to end.
     * @returns {Promise<{accepted:boolean, attempt:object}>}
     */
    async _tryCandidate(backendEntry, model) {
        const attempt = {
            backendId: backendEntry.id,
            backendLabel: BACKEND_LABELS[backendEntry.id] || backendEntry.id,
            modelKey: model.key,
            modelLabel: model.label,
            candidate: model.candidate,
            accepted: false,
            reason: null,
            initMs: null,
            fromCache: null,
            warmup: null
        };

        let adapter = null;
        try {
            // The tensor size is dictated by the MODEL, not by the device
            // profile. The exported ONNX graphs have a static input shape, so
            // feeding a profile-derived 256x256 tensor into a 320x320 graph makes
            // every inference fail. The profile's preferred input size is used to
            // *rank* candidates; the chosen model then sets the actual size.
            if (model.inputSize && model.inputSize !== this.framePipeline.inputSize) {
                this.framePipeline.setInputSize(model.inputSize);
            }
            attempt.inputSize = this.framePipeline.inputSize;

            adapter = this._createAdapter(backendEntry, model);
            this.onEvent("selection-attempt", { ...attempt, stage: "init" });

            const initInfo = await adapter.init();
            attempt.initMs = initInfo.initMs;
            attempt.fromCache = initInfo.fromCache;
            attempt.resolvedBackendId = initInfo.backendId || backendEntry.id;

            this.onEvent("selection-attempt", { ...attempt, stage: "warmup" });

            const warm = await adapter.warmup({
                passes: VisionConfig.warmup.passes,
                letterbox: this.framePipeline.letterbox
            });
            attempt.warmup = warm;

            const verdict = evaluateWarmup(
                warm,
                VisionConfig.warmup.maxAcceptableLatencyMs,
                VisionConfig.warmup.maxFirstPassMs
            );
            attempt.reason = warm.error ? `${verdict.reason} (${warm.error})` : verdict.reason;
            attempt.p50 = verdict.p50;
            attempt.p95 = verdict.p95;
            attempt.firstPassMs = verdict.firstPassMs;

            if (!verdict.accepted) {
                adapter.dispose();
                return { accepted: false, attempt };
            }

            attempt.accepted = true;
            this.adapter = adapter;
            this.backendId = attempt.resolvedBackendId;
            this.model = model;
            this.warmupResult = warm;
            this.initInfo = initInfo;
            this.ready = true;
            return { accepted: true, attempt };
        } catch (err) {
            attempt.reason = String(err?.message || err);
            try {
                adapter?.dispose();
            } catch { /* ignore */ }
            return { accepted: false, attempt };
        }
    }

    /**
     * Select and prepare a detector.
     *
     * @param {{preferredModelKey?:string, preferredBackendId?:string,
     *          restrictToBackendId?:string, restrictToModelKey?:string}} [opts]
     *        The `restrictTo*` forms are used by the benchmark screen, which must
     *        measure one exact (backend, model) pair and report a failure rather
     *        than silently succeeding with a different one.
     * @returns {Promise<{ok:boolean, backendId?:string, model?:object, attempts:object[]}>}
     */
    async initialize({
        preferredModelKey = null,
        preferredBackendId = null,
        restrictToBackendId = null,
        restrictToModelKey = null
    } = {}) {
        this.attempts = [];
        this.ready = false;

        let plan = this.plan;
        if (restrictToBackendId) {
            plan = this.plan.filter((p) => p.id === restrictToBackendId);
            if (plan.length === 0) {
                return { ok: false, attempts: [], reason: `backend ${restrictToBackendId} is not in this device's plan` };
            }
        } else if (preferredBackendId) {
            plan = this.plan.filter((p) => p.id === preferredBackendId)
                .concat(this.plan.filter((p) => p.id !== preferredBackendId));
        }

        for (let i = 0; i < plan.length; i += 1) {
            const backendEntry = plan[i];

            let candidates = rankCandidates({
                backendId: backendEntry.id,
                preferQuantized: this.profile.preferQuantized
            });

            if (restrictToModelKey) {
                const only = getModel(restrictToModelKey);
                if (!only || !only.supportedBackends.includes(backendEntry.id)) continue;
                candidates = [only];
            } else if (preferredModelKey) {
                const preferred = getModel(preferredModelKey);
                if (preferred && preferred.supportedBackends.includes(backendEntry.id)) {
                    candidates = [preferred, ...candidates.filter((c) => c.key !== preferredModelKey)];
                }
            }

            for (const model of candidates) {
                // Bounded per candidate. A single slow session build - WebGPU
                // shader compilation for an attention-heavy graph can take tens of
                // seconds - must not consume the whole startup budget and starve
                // the candidates that would have worked.
                let outcome;
                try {
                    outcome = await withTimeout(
                        this._tryCandidate(backendEntry, model),
                        VisionConfig.warmup.candidateTimeoutMs,
                        `${model.key} on ${backendEntry.id}`
                    );
                } catch (err) {
                    outcome = {
                        accepted: false,
                        attempt: {
                            backendId: backendEntry.id,
                            backendLabel: BACKEND_LABELS[backendEntry.id] || backendEntry.id,
                            modelKey: model.key,
                            modelLabel: model.label,
                            candidate: model.candidate,
                            accepted: false,
                            timedOut: err?.name === "TimeoutError",
                            reason: String(err?.message || err)
                        }
                    };
                    this._disposeAdapter();
                }

                const { accepted, attempt } = outcome;
                this.attempts.push(attempt);
                this.onEvent("selection-result", attempt);

                /**
                 * A timeout indicts the backend, not the model.
                 *
                 * If a session build or warm-up hangs long enough to hit the
                 * budget, trying a second model on the same backend hangs the same
                 * way and costs another full budget. On a device with a broken
                 * WebGPU stack that was two 30-second stalls before reaching the
                 * WASM path — most of a minute of the user staring at a progress
                 * bar. Abandon the backend and move on.
                 */
                if (!accepted && attempt.timedOut) {
                    this.onEvent("backend-abandoned", {
                        backendId: backendEntry.id,
                        backendLabel: BACKEND_LABELS[backendEntry.id] || backendEntry.id,
                        reason: attempt.reason
                    });
                    break;
                }

                if (accepted) {
                    this.planIndex = this.plan.findIndex((p) => p.id === backendEntry.id);
                    return {
                        ok: true,
                        backendId: this.backendId,
                        model: this.model,
                        warmup: this.warmupResult,
                        initInfo: this.initInfo,
                        attempts: this.attempts
                    };
                }
            }
        }

        return { ok: false, attempts: this.attempts };
    }

    /* ---------------------------------------------------------- inference */

    /**
     * Run one inference on the newest available frame.
     *
     * Returns `{skipped:<reason>}` rather than throwing when the pipeline is
     * busy. Skipping is the normal, correct behaviour: we always want the
     * freshest frame, never a backlog.
     *
     * @param {HTMLVideoElement} source
     * @param {number} timestamp
     */
    async detect(source, timestamp) {
        if (!this.ready || !this.adapter) return { skipped: SkipReason.NOT_READY };
        if (this._inFlight) {
            this.skippedBusy += 1;
            return { skipped: SkipReason.BUSY };
        }

        if (!this.adapter.needsTensor) {
            // Main-thread adapters preprocess internally.
            this._inFlight = true;
            this.submitted += 1;
            try {
                const result = await this.adapter.detect({ source, timestamp });
                this.completed += 1;
                this._lastLatencyMs = result.latencyMs;
                return { ...result, timestamp, captureMs: 0 };
            } catch (err) {
                this.failed += 1;
                return { error: String(err?.message || err) };
            } finally {
                this._inFlight = false;
            }
        }

        if (!this.framePipeline.canCapture) return { skipped: SkipReason.NO_BUFFER };

        const frame = this.framePipeline.capture(source);
        if (!frame) return { skipped: SkipReason.NO_FRAME };

        const captureMs = frame.drawMs + frame.convertMs;
        this._inFlight = true;
        this.submitted += 1;

        try {
            const result = await this.adapter.detect({
                buffer: frame.buffer,
                letterbox: frame.letterbox,
                timestamp,
                captureMs
            });
            this.completed += 1;
            this._lastLatencyMs = result.latencyMs;
            return { ...result, captureMs, totalMs: result.latencyMs + captureMs };
        } catch (err) {
            this.failed += 1;
            return { error: String(err?.message || err) };
        } finally {
            this._inFlight = false;
        }
    }

    /* ----------------------------------------------------------- recovery */

    /** Rebuild the current adapter in place. First recovery step. */
    async restart() {
        if (!this.model || !this.backendId) return { ok: false, reason: "nothing to restart" };

        const backendEntry = this.plan.find((p) => p.id === this.backendId)
            || { id: this.backendId, runtime: this.model.runtime };

        this._disposeAdapter();
        const { accepted, attempt } = await this._tryCandidate(backendEntry, this.model);
        this.onEvent("recovery-restart", attempt);
        return { ok: accepted, attempt };
    }

    /**
     * Move to the next backend in the plan. Second recovery step, and the
     * WebGPU -> WASM escape hatch.
     */
    async fallbackToNextBackend() {
        this._disposeAdapter();

        for (let i = this.planIndex + 1; i < this.plan.length; i += 1) {
            const backendEntry = this.plan[i];
            const candidates = rankCandidates({
                backendId: backendEntry.id,
                preferQuantized: true
            });
            for (const model of candidates) {
                const { accepted, attempt } = await this._tryCandidate(backendEntry, model);
                this.attempts.push(attempt);
                this.onEvent("recovery-fallback", attempt);
                if (accepted) {
                    this.planIndex = i;
                    return { ok: true, attempt };
                }
            }
        }
        return { ok: false, reason: "backend plan exhausted" };
    }

    /** Whether another backend remains untried. */
    get hasFallback() {
        return this.planIndex < this.plan.length - 1;
    }

    _disposeAdapter() {
        this.ready = false;
        this._inFlight = false;
        try {
            this.adapter?.dispose();
        } catch { /* ignore */ }
        this.adapter = null;
    }

    /** Push new thresholds or detection limits down to the active adapter. */
    async setDetectionConfig(partial) {
        this.detectionConfig = { ...this.detectionConfig, ...partial };
        await this.adapter?.updateDetectionConfig?.(this.detectionConfig);
    }

    /**
     * Requested inference input size.
     *
     * Refused for tensor-based adapters: the exported ONNX graphs have a static
     * input shape, so changing the tensor size without changing the model would
     * make every inference fail. Reducing resolution on those adapters means
     * selecting a differently-exported model, not resizing the tensor. The
     * scheduler reduces *rate* instead, which is the lever that actually exists.
     */
    async setInputSize(inputSize) {
        if (inputSize === this.framePipeline.inputSize) return { ok: true, changed: false };

        if (this.adapter?.needsTensor) {
            return {
                ok: false,
                changed: false,
                reason: `model ${this.model?.key} has a static ${this.framePipeline.inputSize} input shape`
            };
        }

        this.framePipeline.setInputSize(inputSize);
        return { ok: true, changed: true };
    }

    describe() {
        return {
            ready: this.ready,
            backendId: this.backendId,
            backendLabel: this.backendLabel,
            modelKey: this.model?.key || null,
            modelLabel: this.model?.label || null,
            candidate: this.model?.candidate || null,
            quantization: this.model?.quantization || null,
            inputSize: this.framePipeline.inputSize,
            fromCache: this.initInfo?.fromCache ?? null,
            cachePersisted: this.initInfo?.cachePersisted ?? null,
            modelBytes: this.initInfo?.modelBytes ?? null,
            wasmConfig: this.initInfo?.wasmConfig || null,
            warmupFirstPassMs: this.warmupResult?.firstPassMs ?? null,
            warmupSteadyStateMs: this.warmupResult?.steadyStateMs ?? null,
            submitted: this.submitted,
            completed: this.completed,
            failed: this.failed,
            skippedBusy: this.skippedBusy,
            rejectedOverlaps: this.adapter?.rejectedOverlaps ?? 0,
            hasFallback: this.hasFallback,
            planned: this.plan.map((p) => p.id)
        };
    }

    dispose() {
        this._disposeAdapter();
    }
}
