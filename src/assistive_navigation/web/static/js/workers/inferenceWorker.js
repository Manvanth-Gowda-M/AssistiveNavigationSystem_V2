/**
 * Inference worker (ES module worker).
 *
 * Owns ONNX Runtime Web, the model session, and output decoding. The main
 * thread never blocks on inference; it sends a preprocessed tensor buffer by
 * transfer and gets detections back.
 *
 * Contract with the main thread:
 *   in  init     -> out init-result
 *   in  warmup   -> out warmup-result
 *   in  infer    -> out infer-result | infer-error   (always returns the buffer)
 *   in  config   -> out config-result
 *   in  dispose  -> out disposed
 *
 * Invariants:
 *   - The tensor buffer sent by the main thread is ALWAYS transferred back,
 *     including on the error path. Losing a buffer permanently shrinks the pool
 *     and would slowly starve the scheduler.
 *   - Only one inference is ever in progress. The main thread enforces this;
 *     the worker asserts it and rejects overlap rather than queueing.
 */

import { DetectionDecoder } from "../vision/decoders.js";
import {
    BackendId,
    checkTensorSanity,
    configureOrtEnvironment,
    sessionOptionsFor
} from "../vision/backendSelector.js";
import { loadModelBytes } from "../vision/modelCache.js";
import { DecodeFamily } from "../config/modelRegistry.js";
import { navigationClassIndices } from "../config/classCatalog.js";
import { withTimeout } from "../utils/async.js";

/** Budget for importing the ONNX Runtime bundle from one source. */
const ORT_IMPORT_TIMEOUT_MS = 12000;
/** Budget for fetching the model weights. */
const MODEL_FETCH_TIMEOUT_MS = 90000;
/** Budget for building the inference session once the bytes are in hand. */
const SESSION_CREATE_TIMEOUT_MS = 30000;

/* ------------------------------------------------------------------ state */

let ort = null;
let ortSourceUrl = null;
let session = null;
let modelSpec = null;
let backendId = null;
let wasmConfig = null;
let allowedClassIds = null;
let detectionConfig = {
    scoreThreshold: 0.3,
    nmsIouThreshold: 0.45,
    maxDetections: 24,
    minBoxArea: 0.0015
};

const decoder = new DetectionDecoder();

/** Output dims, recorded on the first run for the diagnostics panel. */
let outputDims = null;
let inferenceInProgress = false;
let inferenceCount = 0;
let failureCount = 0;

/* ------------------------------------------------------------- utilities */

function log(level, message, detail) {
    self.postMessage({ type: "log", level, message, detail });
}

function now() {
    return performance.now();
}

/**
 * Load ONNX Runtime Web.
 *
 * `ort.all` carries every execution provider, which is what we want because the
 * backend is chosen at runtime rather than at build time. Several candidates are
 * tried because this is the one hard dependency on a third party: a CDN outage,
 * a blocked domain or a layout change here would otherwise take the whole app
 * down.
 *
 * Every import is bounded. A dynamic import against an unreachable host can hang
 * for the browser's own connect timeout, which is far longer than a user will
 * wait staring at "Initialising".
 */
async function loadOrt(baseUrl, version) {
    if (ort) return ort;

    const bundles = ["ort.all.min.mjs", "ort.min.mjs", "ort.webgpu.min.mjs", "ort.wasm.min.mjs"];
    const attempts = [];

    /**
     * Same-origin copy first, if one has been vendored into `static/vendor/ort/`.
     *
     * When present this removes the only third-party runtime dependency the app
     * has. When absent the import 404s in milliseconds against our own origin and
     * we move on, so carrying this entry costs nothing.
     */
    const vendorBase = new URL("../../vendor/ort/", import.meta.url).href;
    for (const bundle of bundles) {
        attempts.push({
            url: `${vendorBase}${bundle}`,
            wasmPaths: vendorBase,
            label: `self-hosted/${bundle}`,
            optional: true
        });
    }

    const mirrors = [
        { base: baseUrl, label: "jsdelivr" },
        { base: "https://unpkg.com/", label: "unpkg" }
    ];
    for (const mirror of mirrors) {
        for (const bundle of bundles) {
            attempts.push({
                url: `${mirror.base}onnxruntime-web@${version}/dist/${bundle}`,
                wasmPaths: `${mirror.base}onnxruntime-web@${version}/dist/`,
                label: `${mirror.label}/${bundle}`
            });
        }
    }

    const failures = [];
    for (const attempt of attempts) {
        try {
            const mod = await withTimeout(
                import(/* @vite-ignore */ attempt.url),
                ORT_IMPORT_TIMEOUT_MS,
                `import ${attempt.label}`
            );
            const namespace = mod?.default?.InferenceSession ? mod.default : mod;
            if (!namespace?.InferenceSession) throw new Error("module exposes no InferenceSession");

            ort = namespace;
            ortSourceUrl = attempt.url;
            // The .wasm and .mjs runtime artifacts sit beside the JS bundle.
            ort.env.wasm.wasmPaths = attempt.wasmPaths;
            // Keep ORT quiet; we do our own reporting.
            ort.env.logLevel = "error";
            log("info", `ONNX Runtime loaded from ${attempt.label}`);
            return ort;
        } catch (err) {
            const message = String(err?.message || err);
            failures.push(`${attempt.label}: ${message}`);
            // A missing vendored copy is the expected case, not a problem worth
            // reporting to the user.
            if (!attempt.optional) {
                log("warn", `ONNX Runtime not available from ${attempt.label}`, message);
            }
        }
    }

    throw new Error(
        `Could not load onnxruntime-web from any source. Tried ${attempts.length}: ${failures.join(" | ")}`
    );
}

function disposeTensor(tensor) {
    if (tensor && typeof tensor.dispose === "function") {
        try {
            tensor.dispose();
        } catch { /* already released */ }
    }
}

function releaseSession() {
    outputDims = null;
    if (session && typeof session.release === "function") {
        try {
            session.release();
        } catch { /* ignore */ }
    }
    session = null;
}

/* ------------------------------------------------------------------ init */

async function handleInit(payload) {
    const {
        model,
        modelUrl,
        backendId: requestedBackend,
        capabilities,
        ortBaseUrl = "https://cdn.jsdelivr.net/npm/",
        ortVersion = "1.23.0",
        detection,
        useCache = true
    } = payload;

    const started = now();
    releaseSession();

    modelSpec = model;
    backendId = requestedBackend;
    if (detection) detectionConfig = { ...detectionConfig, ...detection };

    // Decode-time class filter. Keeps irrelevant COCO classes from ever
    // becoming objects.
    const indices = navigationClassIndices(model.classNames);
    allowedClassIds = new Set(indices);

    self.postMessage({ type: "init-stage", stage: "runtime" });
    await loadOrt(ortBaseUrl, ortVersion);
    wasmConfig = configureOrtEnvironment(ort, backendId, capabilities || {});

    self.postMessage({ type: "init-stage", stage: "weights" });
    const loadStarted = now();
    const cacheResult = await withTimeout(
        loadModelBytes({
            key: model.key,
            url: modelUrl,
            expectedBytes: model.approxBytes || 0,
            allowCache: useCache,
            onProgress: (p) => self.postMessage({ type: "model-progress", ...p })
        }),
        MODEL_FETCH_TIMEOUT_MS,
        `fetch ${model.key}`
    );
    const fetchMs = Math.round(now() - loadStarted);

    self.postMessage({ type: "init-stage", stage: "session" });
    const options = sessionOptionsFor(backendId);
    session = await withTimeout(
        ort.InferenceSession.create(new Uint8Array(cacheResult.bytes), options),
        SESSION_CREATE_TIMEOUT_MS,
        `create ${backendId} session`
    );

    inferenceCount = 0;
    failureCount = 0;

    return {
        ok: true,
        backendId,
        modelKey: model.key,
        wasmConfig,
        ortVersion,
        ortSourceUrl,
        inputNames: session.inputNames,
        outputNames: session.outputNames,
        fromCache: cacheResult.fromCache,
        cachePersisted: cacheResult.persisted,
        modelBytes: cacheResult.byteLength,
        fetchMs,
        initMs: Math.round(now() - started),
        allowedClassCount: allowedClassIds.size
    };
}

/* --------------------------------------------------------------- inference */

function inputName() {
    return modelSpec?.inputName && session.inputNames.includes(modelSpec.inputName)
        ? modelSpec.inputName
        : session.inputNames[0];
}

function outputName() {
    return modelSpec?.outputName && session.outputNames.includes(modelSpec.outputName)
        ? modelSpec.outputName
        : session.outputNames[0];
}

/**
 * Run the session once.
 *
 * Deliberately lets ONNX Runtime allocate the output.
 *
 * An earlier version passed a preallocated output tensor via `fetches` to avoid a
 * per-inference allocation. It did not work: the tensor came back untouched, so
 * every pass after the first returned all zeros. The warm-up sanity check caught
 * it - correctly - and rejected every ONNX candidate on every backend, which left
 * the app permanently on the slow main-thread fallback. Detection would have been
 * silently empty in the live pipeline too.
 *
 * The allocation this was trying to save is a fixed 705 KB per inference for the
 * YOLO11n head. That is the same order as the frame's `ImageData` and is
 * comfortably within what the GC handles; producing correct output is not
 * negotiable. If it is ever worth revisiting, it needs a test that asserts the
 * second and third inferences are non-zero.
 */
async function runSession(inputTensor) {
    const outName = outputName();
    const results = await session.run({ [inputName()]: inputTensor });
    const out = results[outName] || results[session.outputNames[0]];
    if (out?.dims && !outputDims) outputDims = out.dims.slice();
    return out;
}

function decodeOutput(output, letterbox) {
    const data = output.data;
    const dims = output.dims;
    const classNames = modelSpec.classNames;

    if (modelSpec.decode === DecodeFamily.YOLO_E2E) {
        return decoder.decodeYoloEndToEnd(data, dims, {
            letterbox,
            classNames,
            scoreThreshold: detectionConfig.scoreThreshold,
            maxDetections: detectionConfig.maxDetections,
            minBoxArea: detectionConfig.minBoxArea,
            allowedClassIds
        });
    }

    return decoder.decodeYoloRaw(data, dims, {
        letterbox,
        classNames,
        scoreThreshold: detectionConfig.scoreThreshold,
        iouThreshold: detectionConfig.nmsIouThreshold,
        maxDetections: detectionConfig.maxDetections,
        minBoxArea: detectionConfig.minBoxArea,
        allowedClassIds
    });
}

async function handleInfer(payload) {
    const { frameId, buffer, inputSize, letterbox, timestamp, captureMs } = payload;

    if (!session) {
        self.postMessage(
            { type: "infer-error", frameId, message: "session not initialised", buffer },
            [buffer]
        );
        return;
    }
    if (inferenceInProgress) {
        // Should be impossible: the client gates on in-flight state. Reject
        // rather than queue, so a protocol bug cannot become latency.
        self.postMessage(
            { type: "infer-error", frameId, message: "overlapping inference rejected", buffer, rejected: true },
            [buffer]
        );
        return;
    }

    inferenceInProgress = true;
    const t0 = now();
    let inputTensor = null;

    try {
        const view = new Float32Array(buffer);
        inputTensor = new ort.Tensor("float32", view, [1, 3, inputSize, inputSize]);

        const runStart = now();
        const output = await runSession(inputTensor);
        const inferenceMs = now() - runStart;

        const decodeStart = now();
        const detections = decodeOutput(output, letterbox);
        const decodeMs = now() - decodeStart;

        disposeTensor(output);

        inferenceCount += 1;
        failureCount = 0;

        self.postMessage(
            {
                type: "infer-result",
                frameId,
                timestamp,
                detections,
                latencyMs: now() - t0,
                inferenceMs,
                decodeMs,
                captureMs,
                inferenceCount,
                buffer
            },
            [buffer]
        );
    } catch (err) {
        failureCount += 1;
        self.postMessage(
            {
                type: "infer-error",
                frameId,
                message: String(err?.message || err),
                failureCount,
                buffer
            },
            [buffer]
        );
    } finally {
        // The input tensor wraps the transferred buffer; dispose only releases
        // ORT-side bookkeeping, it does not free our pooled memory.
        disposeTensor(inputTensor);
        inferenceInProgress = false;
    }
}

/* ----------------------------------------------------------------- warmup */

/**
 * Run N inferences on synthetic input before the user is allowed to walk.
 *
 * The first inference on any backend pays for shader compilation, kernel
 * selection and memory arena growth; on WebGPU that can be several hundred
 * milliseconds. Paying it here rather than during the first three steps of the
 * demo is the whole point (requirement 15).
 */
async function handleWarmup(payload) {
    const { passes = 4, inputSize, letterbox } = payload;
    if (!session) throw new Error("session not initialised");

    const latencies = [];
    const elementCount = inputSize * inputSize * 3;
    const scratch = new Float32Array(elementCount);

    // Mid-grey with a deterministic gradient. A constant frame can let some
    // execution providers take shortcuts that make the measurement optimistic.
    for (let i = 0; i < elementCount; i += 1) {
        scratch[i] = 0.35 + 0.3 * ((i % 251) / 251);
    }

    let producedOutput = false;
    let outputSane = false;
    let lastError = null;

    for (let p = 0; p < passes; p += 1) {
        const tensor = new ort.Tensor("float32", scratch, [1, 3, inputSize, inputSize]);
        const t0 = now();
        try {
            const output = await runSession(tensor);
            latencies.push(now() - t0);
            if (output?.data?.length) {
                producedOutput = true;
                // An end-to-end head legitimately emits all zeros when it finds
                // nothing, which is the expected result for a synthetic frame.
                const sanity = checkTensorSanity(output.data, {
                    expectNonZero: modelSpec.decode !== DecodeFamily.YOLO_E2E
                });
                outputSane = sanity.sane;
                if (!sanity.sane) lastError = sanity.reason;
                // Exercise the decoder too: a decode-time exception during a
                // live walk is exactly what warm-up should catch.
                decodeOutput(output, letterbox);
            }
            disposeTensor(output);
        } catch (err) {
            lastError = String(err?.message || err);
            break;
        } finally {
            disposeTensor(tensor);
        }
    }

    return {
        latencies,
        producedOutput,
        outputSane,
        error: lastError,
        // The first pass is dominated by one-time compilation; report both so
        // the dashboard can show the real steady-state figure.
        firstPassMs: latencies.length ? Math.round(latencies[0]) : 0,
        steadyStateMs: latencies.length > 1
            ? Math.round(latencies.slice(1).reduce((a, b) => a + b, 0) / (latencies.length - 1))
            : 0
    };
}

/* --------------------------------------------------------------- dispatch */

self.onmessage = async (event) => {
    const { type, id, payload } = event.data || {};

    try {
        switch (type) {
            case "init": {
                const result = await handleInit(payload);
                self.postMessage({ type: "init-result", id, ok: true, ...result });
                break;
            }
            case "warmup": {
                const result = await handleWarmup(payload);
                self.postMessage({ type: "warmup-result", id, ok: true, ...result });
                break;
            }
            case "infer":
                await handleInfer(payload);
                break;
            case "config":
                if (payload?.detection) detectionConfig = { ...detectionConfig, ...payload.detection };
                self.postMessage({ type: "config-result", id, ok: true, detectionConfig });
                break;
            case "stats":
                self.postMessage({
                    type: "stats-result",
                    id,
                    ok: true,
                    inferenceCount,
                    failureCount,
                    backendId,
                    modelKey: modelSpec?.key || null,
                    wasmConfig,
                    outputDims
                });
                break;
            case "dispose":
                releaseSession();
                self.postMessage({ type: "disposed", id, ok: true });
                break;
            default:
                self.postMessage({ type: "error", id, ok: false, message: `unknown message: ${type}` });
        }
    } catch (err) {
        self.postMessage({
            type: `${type}-result`,
            id,
            ok: false,
            message: String(err?.message || err),
            stack: err?.stack ? String(err.stack).split("\n").slice(0, 4).join("\n") : null
        });
    }
};

self.postMessage({ type: "worker-ready" });
