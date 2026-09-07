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
 * Three independent gates:
 *   1. It produced output at all (`producedOutput`).
 *   2. Its output was structurally sane (`outputSane`) - a WebGPU driver that
 *      silently emits zeros or NaNs is the classic "available but broken" case.
 *   3. Its p95 latency is inside budget. p95 rather than mean, because a
 *      backend that is fast on average but spikes to 500 ms every tenth frame
 *      makes guidance feel broken.
 *
 * @param {object} result
 * @param {number[]} result.latencies
 * @param {boolean} result.producedOutput
 * @param {boolean} result.outputSane
 * @param {number} maxAcceptableLatencyMs
 * @returns {{accepted:boolean, reason:string, p50:number, p95:number, mean:number}}
 */
export function evaluateWarmup({ latencies = [], producedOutput = false, outputSane = false }, maxAcceptableLatencyMs) {
    const p50 = Math.round(percentile(latencies, 50));
    const p95 = Math.round(percentile(latencies, 95));
    const mean = latencies.length
        ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length)
        : 0;

    if (!producedOutput) {
        return { accepted: false, reason: "no output produced during warm-up", p50, p95, mean };
    }
    if (!outputSane) {
        return { accepted: false, reason: "warm-up output failed the sanity check", p50, p95, mean };
    }
    if (latencies.length === 0) {
        return { accepted: false, reason: "no latency samples", p50, p95, mean };
    }
    if (p95 > maxAcceptableLatencyMs) {
        return {
            accepted: false,
            reason: `p95 ${p95} ms exceeds the ${maxAcceptableLatencyMs} ms warm-up budget`,
            p50,
            p95,
            mean
        };
    }
    return { accepted: true, reason: "passed", p50, p95, mean };
}

/**
 * Structural sanity check on a detector's raw output tensor.
 *
 * We are not checking accuracy here - a warm-up frame is synthetic and should
 * legitimately find nothing. We are checking that the numbers are numbers.
 *
 * @param {ArrayLike<number>} data
 * @returns {{sane:boolean, reason:string}}
 */
export function checkTensorSanity(data) {
    if (!data || data.length === 0) return { sane: false, reason: "empty output tensor" };

    // Sample rather than scan: these tensors have >100k elements and this runs
    // on the critical path to "ready".
    const stride = Math.max(1, Math.floor(data.length / 4096));
    let nonZero = 0;
    let samples = 0;

    for (let i = 0; i < data.length; i += stride) {
        const v = data[i];
        if (!Number.isFinite(v)) return { sane: false, reason: "output contains NaN or Infinity" };
        if (v !== 0) nonZero += 1;
        samples += 1;
    }

    if (samples === 0) return { sane: false, reason: "no samples read" };
    // An all-zero tensor from a real detector head is not physically plausible:
    // box regression rows always carry anchor-derived values.
    if (nonZero === 0) return { sane: false, reason: "output is entirely zero" };

    return { sane: true, reason: "ok" };
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
