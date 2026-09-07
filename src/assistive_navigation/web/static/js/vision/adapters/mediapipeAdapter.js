/**
 * MediaPipe Tasks Vision object-detector adapter (Candidate A: EfficientDet-Lite0).
 *
 * MediaPipe needs a real DOM/GL context, so this runs on the main thread. It is
 * kept as a benchmark candidate and as a recovery path: if the ONNX worker
 * cannot be made to behave on a particular device, the app still detects rather
 * than going blind.
 *
 * Because MediaPipe does its own preprocessing, this adapter consumes the video
 * element directly and emits boxes already normalised to camera space.
 */

import { normaliseNativeDetections } from "../decoders.js";
import { isNavigationRelevant } from "../../config/classCatalog.js";

const TASKS_VISION_VERSION = "0.10.14";
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}`;

export class MediaPipeDetectorAdapter {
    /**
     * @param {{model:object, modelUrl:string, delegate:'GPU'|'CPU', detection:object}} opts
     */
    constructor({ model, modelUrl, delegate = "GPU", detection }) {
        this.model = model;
        this.modelUrl = modelUrl;
        this.delegate = delegate;
        this.detection = detection;

        this.kind = "main";
        /** Consumes a source element, not a preprocessed tensor. */
        this.needsTensor = false;
        this.backendLabel = `MediaPipe ${delegate}`;

        this._detector = null;
        this._raw = [];
        this.inferenceCount = 0;
    }

    async init() {
        const started = performance.now();
        const mod = await import(/* @vite-ignore */ `${CDN_BASE}`);
        const { FilesetResolver, ObjectDetector } = mod;
        if (!FilesetResolver || !ObjectDetector) {
            throw new Error("MediaPipe tasks-vision did not expose ObjectDetector");
        }

        const fileset = await FilesetResolver.forVisionTasks(`${CDN_BASE}/wasm`);
        this._detector = await ObjectDetector.createFromOptions(fileset, {
            baseOptions: {
                modelAssetPath: this.modelUrl,
                delegate: this.delegate
            },
            runningMode: "VIDEO",
            // Threshold at the runtime boundary so irrelevant boxes never reach us.
            scoreThreshold: this.detection.scoreThreshold,
            maxResults: this.detection.maxDetections
        });

        return {
            ok: true,
            backendId: this.delegate === "GPU" ? "mediapipe-gpu" : "mediapipe-cpu",
            modelKey: this.model.key,
            initMs: Math.round(performance.now() - started),
            fromCache: false,
            // MediaPipe fetches and caches the .tflite itself through the HTTP
            // cache; we do not control its storage.
            cachePersisted: false,
            modelBytes: this.model.approxBytes || 0
        };
    }

    /**
     * @param {{source:HTMLVideoElement, timestamp:number}} request
     * @returns {Promise<{detections:Array, latencyMs:number}>}
     */
    async detect({ source, timestamp }) {
        if (!this._detector) throw new Error("MediaPipe detector not initialised");

        const srcW = source.videoWidth || source.width || 0;
        const srcH = source.videoHeight || source.height || 0;
        if (!srcW || !srcH) return { detections: [], latencyMs: 0 };

        const t0 = performance.now();
        // MediaPipe requires strictly increasing timestamps in VIDEO mode.
        const result = this._detector.detectForVideo(source, Math.max(1, Math.round(timestamp)));
        const latencyMs = performance.now() - t0;

        const raw = this._raw;
        raw.length = 0;

        if (result?.detections) {
            for (const det of result.detections) {
                const cat = det.categories?.[0];
                if (!cat || !det.boundingBox) continue;
                const label = String(cat.categoryName || "obstacle").toLowerCase();
                if (!isNavigationRelevant(label)) continue;
                const bb = det.boundingBox;
                raw.push({
                    label,
                    confidence: cat.score,
                    box: [
                        bb.originX / srcW,
                        bb.originY / srcH,
                        (bb.originX + bb.width) / srcW,
                        (bb.originY + bb.height) / srcH
                    ]
                });
            }
        }

        this.inferenceCount += 1;
        return {
            detections: normaliseNativeDetections(raw, this.detection),
            latencyMs,
            inferenceMs: latencyMs,
            decodeMs: 0
        };
    }

    /**
     * Warm up against a synthetic canvas. MediaPipe's first GPU call compiles
     * shaders; doing that here keeps it out of the first live frames.
     */
    async warmup({ passes = 4 }) {
        const canvas = document.createElement("canvas");
        canvas.width = 320;
        canvas.height = 240;
        const ctx = canvas.getContext("2d");
        const gradient = ctx.createLinearGradient(0, 0, 320, 240);
        gradient.addColorStop(0, "#404040");
        gradient.addColorStop(1, "#a0a0a0");
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, 320, 240);

        const latencies = [];
        let producedOutput = false;
        let error = null;

        for (let i = 0; i < passes; i += 1) {
            const t0 = performance.now();
            try {
                const result = this._detector.detectForVideo(canvas, Date.now() + i);
                latencies.push(performance.now() - t0);
                if (result) producedOutput = true;
            } catch (err) {
                error = String(err?.message || err);
                break;
            }
        }

        return {
            latencies,
            producedOutput,
            // MediaPipe returns structured results rather than a raw tensor, so
            // "did it return a result object" is the strongest structural check
            // available here.
            outputSane: producedOutput,
            error,
            firstPassMs: latencies.length ? Math.round(latencies[0]) : 0,
            steadyStateMs: latencies.length > 1
                ? Math.round(latencies.slice(1).reduce((a, b) => a + b, 0) / (latencies.length - 1))
                : 0
        };
    }

    updateDetectionConfig(detection) {
        this.detection = { ...this.detection, ...detection };
    }

    dispose() {
        try {
            this._detector?.close?.();
        } catch { /* ignore */ }
        this._detector = null;
    }
}
