/**
 * Inference backend selection policy.
 *
 * The rule this module exists to enforce: **"WebGPU is available" is not the
 * same as "WebGPU is faster"**. Every backend is attempted, warmed up, measured,
 * and only then accepted. A backend that initialises but produces garbage or
 * 400 ms latencies is rejected and the next one is tried.
 *
 * The policy functions are pure so they can be unit-tested; the ORT environment
 * configuration takes the `ort` namespace as a parameter rather than importing
 * it, so this module stays loadable in Node.
 */

/** Backend identifiers. These match `supportedBackends` in the model registry. */
export const BackendId = Object.freeze({
    WEBGPU: "webgpu",
    WASM_SIMD: "wasm-simd",
    WASM: "wasm",
    MEDIAPIPE_GPU: "mediapipe-gpu",
    MEDIAPIPE_CPU: "mediapipe-cpu",
    TFJS_WEBGL: "tfjs-webgl",
    TFJS_CPU: "tfjs-cpu"
});

/** Which runtime owns each backend. */
export const BACKEND_RUNTIME = Object.freeze({
    [BackendId.WEBGPU]: "ORT",
    [BackendId.WASM_SIMD]: "ORT",
    [BackendId.WASM]: "ORT",
    [BackendId.MEDIAPIPE_GPU]: "MEDIAPIPE",
    [BackendId.MEDIAPIPE_CPU]: "MEDIAPIPE",
    [BackendId.TFJS_WEBGL]: "TFJS",
    [BackendId.TFJS_CPU]: "TFJS"
});

/** Human labels for the dashboard. */
export const BACKEND_LABELS = Object.freeze({
    [BackendId.WEBGPU]: "ONNX Runtime · WebGPU",
    [BackendId.WASM_SIMD]: "ONNX Runtime · WASM SIMD",
    [BackendId.WASM]: "ONNX Runtime · WASM",
    [BackendId.MEDIAPIPE_GPU]: "MediaPipe · GPU delegate",
    [BackendId.MEDIAPIPE_CPU]: "MediaPipe · CPU delegate",
    [BackendId.TFJS_WEBGL]: "TensorFlow.js · WebGL",
    [BackendId.TFJS_CPU]: "TensorFlow.js · CPU"
});

/**
 * Build the ordered list of backends to attempt.
 *
 * Ordering rationale:
 *   - WebGPU first when present, because when it works it is decisively faster.
 *   - WASM SIMD next. On GitHub Pages this is single-threaded (no cross-origin
 *     isolation, so no SharedArrayBuffer) and that is a first-class path here,
 *     not a degraded one.
 *   - Plain WASM as the floor.
 *   - MediaPipe and TFJS are main-thread runtimes retained as recovery paths:
 *     if the ORT worker cannot be made to behave on a device, the app still
 *     detects rather than dying.
 *
 * @param {object} caps result of probeDeviceCapabilities()
 * @param {{allowMainThreadFallback?:boolean}} [opts]
 * @returns {Array<{id:string, runtime:string, reason:string}>}
 */
export function buildBackendPlan(caps = {}, opts = {}) {
    const { allowMainThreadFallback = true } = opts;
    const plan = [];

    const workerCapable = caps.hasWorker !== false;

    if (caps.hasWebGpu && workerCapable) {
        plan.push({
            id: BackendId.WEBGPU,
            runtime: "ORT",
            reason: "WebGPU adapter reported; must still pass the warm-up gate"
        });
    }

    if (workerCapable && caps.hasWasmSimd !== false) {
        plan.push({
            id: BackendId.WASM_SIMD,
            runtime: "ORT",
            reason: caps.hasWasmThreads
                ? "WASM SIMD with threads (cross-origin isolated)"
                : "WASM SIMD, single thread (no cross-origin isolation)"
        });
    }

    if (workerCapable) {
        plan.push({
            id: BackendId.WASM,
            runtime: "ORT",
            reason: "Baseline WASM without SIMD"
        });
    }

    if (allowMainThreadFallback) {
        plan.push({
            id: BackendId.MEDIAPIPE_GPU,
            runtime: "MEDIAPIPE",
            reason: "Main-thread recovery path with GPU delegate"
        });
        plan.push({
            id: BackendId.MEDIAPIPE_CPU,
            runtime: "MEDIAPIPE",
            reason: "Main-thread recovery path with CPU delegate"
        });
        plan.push({
            id: BackendId.TFJS_WEBGL,
            runtime: "TFJS",
            reason: "Last-resort detector that is known to load everywhere"
        });
    }

    return plan;
}

/** Percentile over an unsorted numeric array. Does not mutate the input. */
export function percentile(values, p) {
    if (!values || values.length === 0) return 0;
    const sorted = Float64Array.from(values).sort();
    const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
    return sorted[idx];
}

/**
 * Decide whether a warmed-up backend is fit for live use.
 *
 * Four independent gates:
 *   1. It produced output at all (`producedOutput`).
 *   2. Its output was structurally sane (`outputSane`) - a WebGPU driver that
 *      silently emits zeros or NaNs is the classic "available but broken" case.
 *   3. Its **steady-state** p95 latency is inside budget.
 *   4. Its first pass, which is a separate and much larger budget, completed.
 *
 * The steady-state/first-pass split matters and getting it wrong broke startup
 * completely. The first warm-up pass pays for shader compilation, kernel
 * selection and memory-arena growth; on WebGPU that is routinely hundreds of
 * milliseconds to several seconds. An earlier version took p95 across *all*
 * passes, and with only four samples p95 *is* the maximum - which is always the
 * first pass. Every backend was therefore rejected for its one-time startup cost,
 * on every device, and the app never selected a detector at all.
 *
 * Steady state is what predicts the walking experience. First-pass cost predicts
 * how long the user waits before starting, which is a different question with a
 * different, far more generous budget.
 *
 * @param {object} result
 * @param {number[]} result.latencies in call order; the first is the cold pass
 * @param {boolean} result.producedOutput
 * @param {boolean} result.outputSane
 * @param {number} maxAcceptableLatencyMs steady-state budget
 * @param {number} [maxFirstPassMs] cold-start budget
 * @returns {{accepted:boolean, reason:string, p50:number, p95:number, mean:number, firstPassMs:number}}
 */
export function evaluateWarmup(
    { latencies = [], producedOutput = false, outputSane = false },
    maxAcceptableLatencyMs,
    maxFirstPassMs = 12000
) {
    const firstPassMs = latencies.length ? Math.round(latencies[0]) : 0;
    // Judge everything after the cold pass.
    const steady = latencies.length > 1 ? latencies.slice(1) : latencies;
    const p50 = Math.round(percentile(steady, 50));
    const p95 = Math.round(percentile(steady, 95));
    const mean = steady.length ? Math.round(steady.reduce((a, b) => a + b, 0) / steady.length) : 0;
    const summary = { p50, p95, mean, firstPassMs };

    if (!producedOutput) {
        return { accepted: false, reason: "no output produced during warm-up", ...summary };
    }
    if (!outputSane) {
        return { accepted: false, reason: "warm-up output failed the sanity check", ...summary };
    }
    if (latencies.length < 2) {
        return {
            accepted: false,
            reason: `only ${latencies.length} warm-up pass(es); need at least 2 to measure steady state`,
            ...summary
        };
    }
    if (firstPassMs > maxFirstPassMs) {
        return {
            accepted: false,
            reason: `first pass ${firstPassMs} ms exceeds the ${maxFirstPassMs} ms cold-start budget`,
            ...summary
        };
    }
    if (p95 > maxAcceptableLatencyMs) {
        return {
            accepted: false,
            reason: `steady-state p95 ${p95} ms exceeds the ${maxAcceptableLatencyMs} ms budget`
                + ` (first pass was ${firstPassMs} ms and is excluded)`,
            ...summary
        };
    }
    return { accepted: true, reason: "passed", ...summary };
}

/**
 * Structural sanity check on a detector's raw output tensor.
 *
 * We are not checking accuracy - a warm-up frame is synthetic and should
 * legitimately find nothing. We are checking that the numbers are numbers, and
 * that the backend actually wrote something.
 *
 * The all-zero rule is **head-dependent**, and getting that wrong rejected every
 * ONNX candidate on every backend:
 *
 *   - A raw YOLO head, [1, 4+numClasses, anchors], always carries anchor-derived
 *     box values in rows 0-3 regardless of what is in the image. All zeros means
 *     the backend did not write to the buffer, which is a genuine fault.
 *
 *   - An end-to-end head, [1, maxDet, 6], emits post-NMS detections. Zero
 *     detections is the correct output for a featureless warm-up frame, so all
 *     zeros is expected and must not be treated as a fault.
 *
 * @param {ArrayLike<number>} data
 * @param {{expectNonZero?:boolean}} [opts]
 * @returns {{sane:boolean, reason:string, nonZeroRatio:number}}
 */
export function checkTensorSanity(data, { expectNonZero = true } = {}) {
    if (!data || data.length === 0) {
        return { sane: false, reason: "empty output tensor", nonZeroRatio: 0 };
    }

    // Sample rather than scan: these tensors have >100k elements and this runs
    // on the critical path to "ready".
    const stride = Math.max(1, Math.floor(data.length / 4096));
    let nonZero = 0;
    let samples = 0;

    for (let i = 0; i < data.length; i += stride) {
        const v = data[i];
        if (!Number.isFinite(v)) {
            return { sane: false, reason: "output contains NaN or Infinity", nonZeroRatio: 0 };
        }
        if (v !== 0) nonZero += 1;
        samples += 1;
    }

    if (samples === 0) return { sane: false, reason: "no samples read", nonZeroRatio: 0 };

    const nonZeroRatio = nonZero / samples;
    if (expectNonZero && nonZero === 0) {
        return { sane: false, reason: "output is entirely zero", nonZeroRatio };
    }

    return { sane: true, reason: "ok", nonZeroRatio };
}

/**
 * Apply ONNX Runtime Web environment settings for a backend.
 *
 * Threads are gated on real SharedArrayBuffer availability rather than on
 * `hardwareConcurrency` alone: requesting threads without cross-origin
 * isolation makes ORT fail to initialise instead of silently degrading.
 *
 * @param {object} ort the onnxruntime-web namespace
 * @param {string} backendId
 * @param {object} caps
 * @returns {{numThreads:number, simd:boolean, proxy:boolean}}
 */
export function configureOrtEnvironment(ort, backendId, caps = {}) {
    const wasm = ort?.env?.wasm;
    const applied = { numThreads: 1, simd: false, proxy: false };
    if (!wasm) return applied;

    const threadsUsable = Boolean(caps.hasWasmThreads);
    const cores = Number.isFinite(caps.hardwareConcurrency) ? caps.hardwareConcurrency : 4;

    if (backendId === BackendId.WASM) {
        wasm.simd = false;
        wasm.numThreads = 1;
    } else {
        wasm.simd = backendId === BackendId.WASM_SIMD ? true : wasm.simd !== false;
        // Leave one core for the UI thread and camera pipeline; oversubscribing
        // makes latency less predictable, which is worse than being slightly
        // slower on average.
        wasm.numThreads = threadsUsable ? Math.max(1, Math.min(4, cores - 1)) : 1;
    }

    // We already run inference inside our own worker, so ORT's proxy worker
    // would add a second hop for no benefit.
    wasm.proxy = false;

    applied.numThreads = wasm.numThreads;
    applied.simd = Boolean(wasm.simd);
    applied.proxy = Boolean(wasm.proxy);
    return applied;
}

/** Execution provider list to hand to `ort.InferenceSession.create`. */
export function executionProvidersFor(backendId) {
    switch (backendId) {
        case BackendId.WEBGPU:
            // WASM stays in the list as an in-session fallback for any operator
            // the WebGPU EP cannot handle.
            return ["webgpu", "wasm"];
        case BackendId.WASM_SIMD:
        case BackendId.WASM:
        default:
            return ["wasm"];
    }
}

/** Session options tuned for low, predictable latency rather than throughput. */
export function sessionOptionsFor(backendId) {
    return {
        executionProviders: executionProvidersFor(backendId),
        graphOptimizationLevel: "all",
        // Sequential execution keeps per-inference latency variance low, which
        // matters more here than peak throughput.
        executionMode: "sequential",
        enableCpuMemArena: true,
        enableMemPattern: true,
        preferredOutputLocation: backendId === BackendId.WEBGPU ? undefined : undefined
    };
}
