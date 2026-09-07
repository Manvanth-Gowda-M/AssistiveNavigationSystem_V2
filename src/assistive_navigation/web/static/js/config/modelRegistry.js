/**
 * Detector model registry.
 *
 * The application never hard-codes "the best model". It carries several
 * candidates with declared cost/quality profiles, and the benchmark screen
 * (ui/benchmarkScreen.js) measures them on the actual device. `expectedLatency`
 * is only used to order the candidates for a first attempt and to warn when a
 * measurement is wildly worse than advertised - it never substitutes for a
 * measurement (requirement 3).
 *
 * Pure data + pure helpers. Importable in Node.
 */

import { COCO_CLASSES } from "./classCatalog.js";

/** Which inference runtime drives a candidate. */
export const Runtime = Object.freeze({
    /** ONNX Runtime Web, in a worker. WebGPU or WASM execution provider. */
    ORT: "ORT",
    /** MediaPipe Tasks Vision ObjectDetector, main thread (needs DOM/GL). */
    MEDIAPIPE: "MEDIAPIPE",
    /** TensorFlow.js COCO-SSD, main thread. Guaranteed-working fallback. */
    TFJS: "TFJS"
});

/** Output tensor decode families. See vision/decoders.js. */
export const DecodeFamily = Object.freeze({
    /** [1, 4+numClasses, numAnchors]; cx,cy,w,h in input pixels. Needs NMS. */
    YOLO_RAW: "YOLO_RAW",
    /** [1, maxDet, 6]; x1,y1,x2,y2,score,classId. Already NMS-free. */
    YOLO_E2E: "YOLO_E2E",
    /** Runtime returns labelled boxes directly; no tensor decode. */
    NATIVE: "NATIVE"
});

const MEDIAPIPE_BASE = "https://storage.googleapis.com/mediapipe-models/object_detector";

/**
 * Registry entries, ordered by first-attempt preference.
 *
 * accuracyProfile / memoryProfile are coarse labels for the dashboard, not
 * measurements.
 */
export const MODEL_REGISTRY = Object.freeze([
    {
        key: "yolo11n-320-uint8",
        label: "YOLO11n 320 uint8",
        candidate: "B",
        runtime: Runtime.ORT,
        decode: DecodeFamily.YOLO_RAW,
        /** Same-origin, so it is cacheable in IndexedDB and works offline. */
        assetPath: "models/yolo11n_320_uint8.onnx",
        approxBytes: 2_906_924,
        inputSize: 320,
        inputName: "images",
        outputName: "output0",
        layout: "NCHW",
        quantization: "uint8-dynamic",
        classNames: COCO_CLASSES,
        supportedBackends: ["wasm-simd", "wasm"],
        expectedLatencyMs: { "webgpu": 45, "wasm-simd": 90, "wasm": 220 },
        accuracyProfile: "good",
        memoryProfile: "low",
        notes: "uint8 weights are the ONNX Runtime recommendation for the CPU/WASM path."
    },
    {
        key: "yolo11n-320-fp32",
        label: "YOLO11n 320 fp32",
        candidate: "B",
        runtime: Runtime.ORT,
        decode: DecodeFamily.YOLO_RAW,
        assetPath: "models/yolo11n_320.onnx",
        approxBytes: 10_575_770,
        inputSize: 320,
        inputName: "images",
        outputName: "output0",
        layout: "NCHW",
        quantization: "fp32",
        classNames: COCO_CLASSES,
        supportedBackends: ["webgpu", "wasm-simd"],
        expectedLatencyMs: { "webgpu": 28, "wasm-simd": 210, "wasm": 600 },
        accuracyProfile: "best",
        memoryProfile: "medium",
        notes: "Preferred on the WebGPU path, where quantised weights give no benefit."
    },
    {
        key: "yolo26n-320-uint8",
        label: "YOLO26n 320 uint8",
        candidate: "C",
        runtime: Runtime.ORT,
        decode: DecodeFamily.YOLO_E2E,
        assetPath: "models/yolo26n_320_uint8.onnx",
        approxBytes: 2_761_512,
        inputSize: 320,
        inputName: "images",
        outputName: "output0",
        layout: "NCHW",
        quantization: "uint8-dynamic",
        classNames: COCO_CLASSES,
        maxDetections: 300,
        supportedBackends: ["wasm-simd", "wasm"],
        expectedLatencyMs: { "webgpu": 42, "wasm-simd": 95, "wasm": 240 },
        accuracyProfile: "good",
        memoryProfile: "low",
        notes: "End-to-end head: emits xyxy/score/class directly, so no browser-side NMS."
    },
    {
        key: "yolo26n-320-fp32",
        label: "YOLO26n 320 fp32",
        candidate: "C",
        runtime: Runtime.ORT,
        decode: DecodeFamily.YOLO_E2E,
        assetPath: "models/yolo26n_320.onnx",
        approxBytes: 9_767_727,
        inputSize: 320,
        inputName: "images",
        outputName: "output0",
        layout: "NCHW",
        quantization: "fp32",
        classNames: COCO_CLASSES,
        maxDetections: 300,
        supportedBackends: ["webgpu", "wasm-simd"],
        /**
         * Ranked behind YOLO11n fp32 on WebGPU on purpose. The attention blocks in
         * this graph make shader compilation very expensive - measured at over 20
         * seconds of session-build time in headless Chromium - and a candidate that
         * takes half a minute to become ready is a poor first thing to try even if
         * its steady-state latency is excellent. The benchmark screen still
         * measures it properly.
         */
        expectedLatencyMs: { "webgpu": 32, "wasm-simd": 220, "wasm": 620 },
        accuracyProfile: "best",
        memoryProfile: "medium",
        notes: "End-to-end head, but attention blocks make WebGPU session build slow. Benchmark before trusting."
    },
    {
        key: "efficientdet-lite0-int8",
        label: "EfficientDet-Lite0 INT8",
        candidate: "A",
        runtime: Runtime.MEDIAPIPE,
        decode: DecodeFamily.NATIVE,
        assetUrl: `${MEDIAPIPE_BASE}/efficientdet_lite0/int8/1/efficientdet_lite0.tflite`,
        approxBytes: 4_602_880,
        inputSize: 320,
        quantization: "int8",
        classNames: COCO_CLASSES,
        supportedBackends: ["mediapipe-gpu", "mediapipe-cpu"],
        expectedLatencyMs: { "mediapipe-gpu": 35, "mediapipe-cpu": 110 },
        accuracyProfile: "fair",
        memoryProfile: "low",
        crossOrigin: true,
        notes: "Runs on the main thread via MediaPipe Tasks Vision; GPU delegate where available."
    },
    {
        key: "efficientdet-lite0-fp16",
        label: "EfficientDet-Lite0 FP16",
        candidate: "A",
        runtime: Runtime.MEDIAPIPE,
        decode: DecodeFamily.NATIVE,
        assetUrl: `${MEDIAPIPE_BASE}/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
        approxBytes: 7_254_016,
        inputSize: 320,
        quantization: "fp16",
        classNames: COCO_CLASSES,
        supportedBackends: ["mediapipe-gpu", "mediapipe-cpu"],
        expectedLatencyMs: { "mediapipe-gpu": 38, "mediapipe-cpu": 190 },
        accuracyProfile: "good",
        memoryProfile: "medium",
        crossOrigin: true
    },
    {
        key: "cocossd-lite-mobilenet-v2",
        label: "COCO-SSD lite MobileNetV2",
        candidate: "fallback",
        runtime: Runtime.TFJS,
        decode: DecodeFamily.NATIVE,
        assetUrl: null,
        approxBytes: 13_000_000,
        inputSize: 300,
        quantization: "fp32",
        classNames: COCO_CLASSES,
        supportedBackends: ["tfjs-webgl", "tfjs-cpu"],
        expectedLatencyMs: { "tfjs-webgl": 70, "tfjs-cpu": 400 },
        accuracyProfile: "fair",
        memoryProfile: "high",
        crossOrigin: true,
        isLastResort: true,
        notes: "Retained from the pre-upgrade build as the always-works safety net."
    }
]);

const BY_KEY = new Map(MODEL_REGISTRY.map((m) => [m.key, m]));

/** @returns {object|undefined} */
export function getModel(key) {
    return BY_KEY.get(key);
}

/** Absolute URL for a model's bytes, resolved against the app's base URL. */
export function resolveModelUrl(model, baseUrl) {
    if (!model) return null;
    if (model.assetUrl) return model.assetUrl;
    if (!model.assetPath) return null;
    const base = baseUrl || (typeof location !== "undefined" ? location.href : "https://localhost/");
    return new URL(model.assetPath, base).href;
}

/**
 * Candidates a given backend can actually execute.
 * @param {string} backendId e.g. "webgpu", "wasm-simd"
 */
export function modelsForBackend(backendId) {
    return MODEL_REGISTRY.filter((m) => m.supportedBackends.includes(backendId));
}

/**
 * Order candidates for a first attempt on a freshly profiled device.
 *
 * The ordering rule is "cheapest thing that is likely to hold real time",
 * because a model that is 3 mAP better but stalls is worse than useless here.
 * Quantised candidates win on the WASM path; fp32 wins on WebGPU.
 *
 * @param {{backendId:string, preferQuantized:boolean, allowCrossOrigin?:boolean}} opts
 * @returns {object[]}
 */
export function rankCandidates({ backendId, preferQuantized = true, allowCrossOrigin = true } = {}) {
    const usable = MODEL_REGISTRY.filter((m) => {
        if (backendId && !m.supportedBackends.includes(backendId)) return false;
        if (!allowCrossOrigin && m.crossOrigin) return false;
        return true;
    });

    /**
     * Quantisation preference is a property of the *backend*, not of the device
     * tier.
     *
     * On WebGPU, fp32 weights are the right choice: the GPU does the work and
     * quantised weights buy nothing. On any CPU path, fp32 is both several times
     * slower and several times larger — YOLO11n is 10.6 MB fp32 against 2.9 MB
     * uint8. A high-tier device with no working WebGPU was therefore downloading
     * 10 MB to run the slower option, which added most of a minute to first load.
     */
    const quantisedPreferred = backendId === "webgpu" ? false : (preferQuantized || true);

    return usable.slice().sort((a, b) => {
        // Last-resort candidates always sort last.
        if (Boolean(a.isLastResort) !== Boolean(b.isLastResort)) return a.isLastResort ? 1 : -1;

        const aQuant = a.quantization !== "fp32";
        const bQuant = b.quantization !== "fp32";
        if (aQuant !== bQuant) return quantisedPreferred === aQuant ? -1 : 1;

        const aLat = (a.expectedLatencyMs && a.expectedLatencyMs[backendId]) ?? Number.MAX_SAFE_INTEGER;
        const bLat = (b.expectedLatencyMs && b.expectedLatencyMs[backendId]) ?? Number.MAX_SAFE_INTEGER;
        if (aLat !== bLat) return aLat - bLat;

        // Same-origin assets beat CDN assets: cacheable and no third-party dependency.
        if (Boolean(a.crossOrigin) !== Boolean(b.crossOrigin)) return a.crossOrigin ? 1 : -1;
        return a.approxBytes - b.approxBytes;
    });
}

/**
 * The three benchmark candidates named in the engineering brief, plus the
 * fallback, deduplicated by candidate letter. Used to populate the benchmark
 * screen's default run list.
 */
export function benchmarkCandidateKeys() {
    const seen = new Set();
    const keys = [];
    for (const m of MODEL_REGISTRY) {
        if (seen.has(m.candidate)) continue;
        seen.add(m.candidate);
        keys.push(m.key);
    }
    return keys;
}

export default MODEL_REGISTRY;
