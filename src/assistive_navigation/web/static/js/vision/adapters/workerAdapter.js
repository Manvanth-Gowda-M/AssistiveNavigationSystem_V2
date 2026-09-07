/**
 * Worker-backed ONNX Runtime adapter.
 *
 * Wraps `workers/inferenceWorker.js` in a promise API and guarantees the two
 * properties the rest of the system depends on:
 *
 *   1. **Exactly one inference in flight.** `detect()` rejects rather than
 *      queues if called while busy. There is no frame queue anywhere in this
 *      system by design (requirement 9).
 *   2. **Buffers always come home.** The pooled tensor buffer is transferred to
 *      the worker and transferred back on both the success and error paths, so
 *      the pool cannot leak entries and starve the scheduler.
 */

const WORKER_URL = new URL("../../workers/inferenceWorker.js", import.meta.url);

const DEFAULT_TIMEOUTS = {
    ready: 15000,
    init: 45000,
    warmup: 30000,
    infer: 5000
};

export class WorkerDetectorAdapter {
    /**
     * @param {{model:object, modelUrl:string, backendId:string, capabilities:object,
     *          detection:object, inputSize:number, onBufferReturn:(b:ArrayBuffer)=>void,
     *          onLog?:Function, onCrash?:Function, timeouts?:object}} opts
     */
    constructor(opts) {
        this.model = opts.model;
        this.modelUrl = opts.modelUrl;
        this.backendId = opts.backendId;
        this.capabilities = opts.capabilities || {};
        this.detection = opts.detection;
        this.inputSize = opts.inputSize;
        this.onBufferReturn = opts.onBufferReturn || (() => {});
        this.onLog = opts.onLog || (() => {});
        this.onCrash = opts.onCrash || (() => {});
        this.onModelProgress = opts.onModelProgress || (() => {});
        this.timeouts = { ...DEFAULT_TIMEOUTS, ...(opts.timeouts || {}) };

        this.kind = "worker";
        /** Consumes a preprocessed tensor buffer. */
        this.needsTensor = true;
        this.backendLabel = `ONNX Runtime ${this.backendId}`;

        this._worker = null;
        this._nextId = 1;
        this._pending = new Map();
        this._readyPromise = null;
        this._inFlight = null;
        this._nextFrameId = 1;
        this._disposed = false;

        this.inferenceCount = 0;
        this.rejectedOverlaps = 0;
    }

    get busy() {
        return this._inFlight !== null;
    }

    /* --------------------------------------------------------------- plumbing */

    _spawn() {
        this._worker = new Worker(WORKER_URL, { type: "module", name: "vision-inference" });

        this._readyPromise = new Promise((resolve, reject) => {
            const timer = setTimeout(
                () => reject(new Error("inference worker did not report ready")),
                this.timeouts.ready
            );
            this._resolveReady = () => {
                clearTimeout(timer);
                resolve();
            };
            this._rejectReady = (err) => {
                clearTimeout(timer);
                reject(err);
            };
        });

        this._worker.onmessage = (event) => this._onMessage(event.data);
        this._worker.onerror = (event) => {
            const message = event?.message || "worker error";
            this._failAll(new Error(message));
            this._rejectReady?.(new Error(message));
            if (!this._disposed) this.onCrash({ reason: "worker-error", message });
        };
        this._worker.onmessageerror = () => {
            this._failAll(new Error("worker message deserialisation failed"));
        };
    }

    _onMessage(msg) {
        if (!msg) return;

        switch (msg.type) {
            case "worker-ready":
                this._resolveReady?.();
                return;
            case "log":
                this.onLog(msg.level, msg.message, msg.detail);
                return;
            case "model-progress":
                this.onModelProgress(msg);
                return;
            case "init-stage":
                // Which part of initialisation the worker reached. Surfaced so a
                // stall can be attributed to the runtime, the download or the
                // session build rather than guessed at.
                this.onModelProgress({ stage: `worker-${msg.stage}` });
                return;
            case "infer-result":
                this._settleInference(msg, null);
                return;
            case "infer-error":
                this._settleInference(msg, new Error(msg.message));
                return;
            default:
                break;
        }

        // Request/response messages carry the correlation id.
        if (msg.id && this._pending.has(msg.id)) {
            const entry = this._pending.get(msg.id);
            this._pending.delete(msg.id);
            clearTimeout(entry.timer);
            if (msg.ok === false) entry.reject(new Error(msg.message || "worker call failed"));
            else entry.resolve(msg);
        }
    }

    _settleInference(msg, error) {
        // Return the buffer before anything else can throw.
        if (msg.buffer) this.onBufferReturn(msg.buffer);

        const inFlight = this._inFlight;
        this._inFlight = null;

        if (!inFlight) return;
        if (inFlight.timer) clearTimeout(inFlight.timer);

        if (msg.rejected) this.rejectedOverlaps += 1;

        if (error) {
            inFlight.reject(error);
            return;
        }

        this.inferenceCount += 1;
        inFlight.resolve({
            detections: msg.detections,
            latencyMs: msg.latencyMs,
            inferenceMs: msg.inferenceMs,
            decodeMs: msg.decodeMs,
            timestamp: msg.timestamp,
            frameId: msg.frameId
        });
    }

    _failAll(error) {
        for (const [, entry] of this._pending) {
            clearTimeout(entry.timer);
            entry.reject(error);
        }
        this._pending.clear();

        if (this._inFlight) {
            const inFlight = this._inFlight;
            this._inFlight = null;
            clearTimeout(inFlight.timer);
            // The buffer went down with the worker; tell the pool so it can
            // account for the loss instead of waiting forever.
            this.onBufferReturn(null);
            inFlight.reject(error);
        }
    }

    _call(type, payload, timeoutMs) {
        const id = this._nextId++;
        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this._pending.delete(id);
                reject(new Error(`worker call '${type}' timed out after ${timeoutMs} ms`));
            }, timeoutMs);
            this._pending.set(id, { resolve, reject, timer });
            this._worker.postMessage({ type, id, payload });
        });
    }

    /* ------------------------------------------------------------- lifecycle */

    async init() {
        this._spawn();
        await this._readyPromise;

        const result = await this._call("init", {
            model: this.model,
            modelUrl: this.modelUrl,
            backendId: this.backendId,
            capabilities: this.capabilities,
            detection: this.detection,
            useCache: true
        }, this.timeouts.init);

        this.backendLabel = `ONNX Runtime ${this.backendId}`;
        return result;
    }

    async warmup({ passes, letterbox }) {
        const result = await this._call("warmup", {
            passes,
            inputSize: this.inputSize,
            letterbox
        }, this.timeouts.warmup);
        return result;
    }

    /**
     * @param {{buffer:Float32Array, letterbox:object, timestamp:number, captureMs:number}} request
     * @returns {Promise<{detections:Array, latencyMs:number}>}
     */
    detect({ buffer, letterbox, timestamp, captureMs }) {
        if (!this._worker) return Promise.reject(new Error("worker not running"));
        if (this._inFlight) {
            this.rejectedOverlaps += 1;
            return Promise.reject(new Error("inference already in flight"));
        }

        const frameId = this._nextFrameId++;
        const arrayBuffer = buffer.buffer;

        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                if (this._inFlight?.frameId !== frameId) return;
                this._inFlight = null;
                // The buffer is stuck inside a wedged worker. Account for the
                // loss; the watchdog will restart the worker.
                this.onBufferReturn(null);
                reject(new Error(`inference timed out after ${this.timeouts.infer} ms`));
            }, this.timeouts.infer);

            this._inFlight = { frameId, resolve, reject, timer };

            this._worker.postMessage({
                type: "infer",
                payload: {
                    frameId,
                    buffer: arrayBuffer,
                    inputSize: this.inputSize,
                    letterbox: {
                        scale: letterbox.scale,
                        padX: letterbox.padX,
                        padY: letterbox.padY,
                        srcW: letterbox.srcW,
                        srcH: letterbox.srcH,
                        dstW: letterbox.dstW,
                        dstH: letterbox.dstH
                    },
                    timestamp,
                    captureMs
                }
            }, [arrayBuffer]);
        });
    }

    async updateDetectionConfig(detection) {
        this.detection = { ...this.detection, ...detection };
        if (!this._worker) return;
        try {
            await this._call("config", { detection: this.detection }, 3000);
        } catch { /* non-fatal */ }
    }

    async stats() {
        if (!this._worker) return null;
        try {
            return await this._call("stats", {}, 2000);
        } catch {
            return null;
        }
    }

    dispose() {
        this._disposed = true;
        this._failAll(new Error("adapter disposed"));
        if (this._worker) {
            try {
                this._worker.postMessage({ type: "dispose" });
            } catch { /* ignore */ }
            // Terminate immediately rather than waiting for a graceful reply:
            // dispose is called on the recovery path where the worker may be
            // wedged.
            this._worker.terminate();
            this._worker = null;
        }
    }
}
