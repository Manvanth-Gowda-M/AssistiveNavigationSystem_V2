/**
 * TensorFlow.js COCO-SSD adapter.
 *
 * This is the detector the pre-upgrade build shipped. It is kept deliberately:
 * it is the one path that has been observed to load on every device we have
 * tested, so it is the floor of the degradation ladder. It is NOT the preferred
 * path - it is heavier and less accurate than the nano ONNX detectors - but a
 * working detector beats a broken one.
 *
 * Loaded lazily from the CDN only if it is actually needed, so the ordinary
 * startup path does not pay for it.
 */

import { normaliseNativeDetections } from "../decoders.js";
import { isNavigationRelevant } from "../../config/classCatalog.js";

const TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js";
const COCOSSD_URL = "https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js";

function loadClassicScript(url) {
    return new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[data-src="${url}"]`);
        if (existing) {
            if (existing.dataset.loaded === "true") resolve();
            else {
                existing.addEventListener("load", () => resolve());
                existing.addEventListener("error", () => reject(new Error(`failed to load ${url}`)));
            }
            return;
        }
        const script = document.createElement("script");
        script.src = url;
        script.async = true;
        script.dataset.src = url;
        script.addEventListener("load", () => {
            script.dataset.loaded = "true";
            resolve();
        });
        script.addEventListener("error", () => reject(new Error(`failed to load ${url}`)));
        document.head.appendChild(script);
    });
}

export class TfjsCocoSsdAdapter {
    constructor({ model, detection, preferWebgl = true }) {
        this.model = model;
        this.detection = detection;
        this.preferWebgl = preferWebgl;

        this.kind = "main";
        this.needsTensor = false;
        this.backendLabel = "TensorFlow.js";

        this._model = null;
        this._tf = null;
        this._raw = [];
        this.inferenceCount = 0;
    }

    async init() {
        const started = performance.now();

        if (!window.tf) await loadClassicScript(TFJS_URL);
        if (!window.cocoSsd) await loadClassicScript(COCOSSD_URL);
        if (!window.cocoSsd) throw new Error("coco-ssd failed to load");

        this._tf = window.tf;
        if (this._tf?.setBackend) {
            try {
                await this._tf.setBackend(this.preferWebgl ? "webgl" : "cpu");
                await this._tf.ready();
            } catch {
                await this._tf.setBackend("cpu");
                await this._tf.ready();
            }
        }

        this._model = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });
        const backend = this._tf?.getBackend?.() || "unknown";
        this.backendLabel = `TensorFlow.js ${backend}`;

        return {
            ok: true,
            backendId: backend === "webgl" ? "tfjs-webgl" : "tfjs-cpu",
            modelKey: this.model.key,
            initMs: Math.round(performance.now() - started),
            fromCache: false,
            cachePersisted: false,
            modelBytes: this.model.approxBytes || 0
        };
    }

    async detect({ source }) {
        if (!this._model) throw new Error("COCO-SSD not initialised");

        const srcW = source.videoWidth || source.width || 0;
        const srcH = source.videoHeight || source.height || 0;
        if (!srcW || !srcH) return { detections: [], latencyMs: 0 };

        const t0 = performance.now();
        const predictions = await this._model.detect(
            source,
            this.detection.maxDetections,
            this.detection.scoreThreshold
        );
        const latencyMs = performance.now() - t0;

        const raw = this._raw;
        raw.length = 0;

        for (const pred of predictions || []) {
            const label = String(pred.class || "obstacle").toLowerCase();
            if (!isNavigationRelevant(label)) continue;
            const [x, y, w, h] = pred.bbox;
            raw.push({
                label,
                confidence: pred.score,
                box: [x / srcW, y / srcH, (x + w) / srcW, (y + h) / srcH]
            });
        }

        this.inferenceCount += 1;
        return {
            detections: normaliseNativeDetections(raw, this.detection),
            latencyMs,
            inferenceMs: latencyMs,
            decodeMs: 0
        };
    }

    async warmup({ passes = 3 }) {
        const canvas = document.createElement("canvas");
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#555";
        ctx.fillRect(0, 0, 320, 240);
        ctx.fillStyle = "#bbb";
        ctx.fillRect(120, 60, 80, 150);

        const latencies = [];
        let producedOutput = false;
        let error = null;

        for (let i = 0; i < passes; i += 1) {
            const t0 = performance.now();
            try {
                const result = await this._model.detect(canvas, 8, 0.3);
                latencies.push(performance.now() - t0);
                if (Array.isArray(result)) producedOutput = true;
            } catch (err) {
                error = String(err?.message || err);
                break;
            }
        }

        return {
            latencies,
            producedOutput,
            outputSane: producedOutput,
            error,
            firstPassMs: latencies.length ? Math.round(latencies[0]) : 0,
            steadyStateMs: latencies.length > 1
                ? Math.round(latencies.slice(1).reduce((a, b) => a + b, 0) / (latencies.length - 1))
                : 0
        };
    }

    /** Tensor accounting, surfaced on the dashboard to catch leaks early. */
    memoryStats() {
        if (!this._tf?.memory) return null;
        const m = this._tf.memory();
        return { numTensors: m.numTensors, numBytes: m.numBytes };
    }

    updateDetectionConfig(detection) {
        this.detection = { ...this.detection, ...detection };
    }

    dispose() {
        try {
            this._model?.dispose?.();
        } catch { /* ignore */ }
        this._model = null;
    }
}
