/**
 * Device profiling and performance profiles.
 *
 * Two halves, deliberately separated so the classification logic is testable
 * without a browser:
 *   - `classifyDeviceTier()` / `resolveProfile()` are pure functions.
 *   - `probeDeviceCapabilities()` touches `navigator` and is browser-only.
 *
 * Profiles are *starting points*. The frame scheduler measures the real device
 * and moves the operating point from here (requirement 61: benchmarked and
 * configurable, not hard-coded truth).
 */

import { VisionConfig } from "./visionConfig.js";

/** Coarse hardware tiers. */
export const DeviceTier = Object.freeze({
    LOW: "LOW",
    MID: "MID",
    HIGH: "HIGH"
});

/** User/auto selectable quality modes. */
export const QualityMode = Object.freeze({
    FAST: "FAST",
    BALANCED: "BALANCED",
    ACCURACY: "ACCURACY"
});

/**
 * Per-tier defaults. `inputSize` and `targetFps` seed the scheduler;
 * `allowSegmentation` gates the optional free-space segmentation stage
 * (requirement 22: never force it onto a device it hurts).
 */
export const TIER_DEFAULTS = Object.freeze({
    [DeviceTier.LOW]: Object.freeze({
        inputSize: 256,
        targetFps: 8,
        maxFps: 12,
        maxDetections: 12,
        allowSegmentation: false,
        preferQuantized: true,
        defaultMode: QualityMode.FAST,
        previewHeight: 480
    }),
    [DeviceTier.MID]: Object.freeze({
        inputSize: 320,
        targetFps: 12,
        maxFps: 18,
        maxDetections: 20,
        allowSegmentation: true,
        preferQuantized: true,
        defaultMode: QualityMode.BALANCED,
        previewHeight: 720
    }),
    [DeviceTier.HIGH]: Object.freeze({
        inputSize: 320,
        targetFps: 16,
        maxFps: 22,
        maxDetections: 24,
        allowSegmentation: true,
        preferQuantized: false,
        defaultMode: QualityMode.BALANCED,
        previewHeight: 720
    })
});

/**
 * Mode deltas applied on top of the tier defaults.
 *
 * ACCURACY is intentionally conservative: requirement 60 warns that accuracy
 * mode must not become so heavy that real-time behaviour breaks, so it raises
 * input size by one step and lowers the FPS target rather than doing both
 * aggressively.
 */
const MODE_ADJUSTMENTS = Object.freeze({
    [QualityMode.FAST]: { inputSizeStep: -1, fpsScale: 1.15, allowSegmentation: false },
    [QualityMode.BALANCED]: { inputSizeStep: 0, fpsScale: 1.0, allowSegmentation: null },
    [QualityMode.ACCURACY]: { inputSizeStep: +1, fpsScale: 0.75, allowSegmentation: null }
});

function stepInputSize(size, step) {
    const sizes = VisionConfig.inference.allowedInputSizes;
    const idx = sizes.indexOf(size);
    const base = idx === -1 ? sizes.indexOf(VisionConfig.inference.defaultInputSize) : idx;
    const next = Math.min(sizes.length - 1, Math.max(0, base + step));
    return sizes[next];
}

/**
 * Classify a device from its reported capabilities.
 *
 * Deliberately pessimistic. `hardwareConcurrency` is the most portable signal
 * we have; `deviceMemory` is Chromium-only and absent on iOS. A device that
 * reports nothing useful lands in MID, not HIGH, because guessing high and
 * thermally collapsing mid-demo is the worse failure.
 *
 * @param {{hardwareConcurrency?:number, deviceMemoryGb?:number, hasWebGpu?:boolean,
 *          screenPixels?:number, isIos?:boolean, userAgentMobile?:boolean}} caps
 * @returns {string} DeviceTier
 */
export function classifyDeviceTier(caps = {}) {
    const cores = Number.isFinite(caps.hardwareConcurrency) ? caps.hardwareConcurrency : 4;
    const memory = Number.isFinite(caps.deviceMemoryGb) ? caps.deviceMemoryGb : null;

    let score = 0;

    if (cores >= 8) score += 2;
    else if (cores >= 6) score += 1;
    else if (cores <= 3) score -= 2;

    if (memory !== null) {
        if (memory >= 8) score += 2;
        else if (memory >= 6) score += 1;
        else if (memory <= 3) score -= 2;
    }

    // WebGPU on a phone browser implies a reasonably current OS and GPU stack.
    if (caps.hasWebGpu) score += 1;

    // iOS devices with WebGPU are consistently strong; without it, Safari's
    // WASM performance is still respectable, so never push iOS to LOW on the
    // core count alone.
    if (caps.isIos) score += 1;

    if (Number.isFinite(caps.screenPixels) && caps.screenPixels < 800 * 480) score -= 1;

    if (score >= 3) return DeviceTier.HIGH;
    if (score <= -2) return DeviceTier.LOW;
    return DeviceTier.MID;
}

/**
 * Resolve the concrete operating parameters for a tier + mode.
 * @param {string} tier DeviceTier
 * @param {string} mode QualityMode
 * @returns {object} frozen profile
 */
export function resolveProfile(tier = DeviceTier.MID, mode = null) {
    const base = TIER_DEFAULTS[tier] || TIER_DEFAULTS[DeviceTier.MID];
    const effectiveMode = mode || base.defaultMode;
    const adj = MODE_ADJUSTMENTS[effectiveMode] || MODE_ADJUSTMENTS[QualityMode.BALANCED];

    const inputSize = stepInputSize(base.inputSize, adj.inputSizeStep);
    const targetFps = Math.round(
        Math.min(VisionConfig.scheduler.maxFps, Math.max(VisionConfig.scheduler.minFps, base.targetFps * adj.fpsScale))
    );
    const maxFps = Math.round(
        Math.min(VisionConfig.scheduler.maxFps, Math.max(targetFps, base.maxFps * adj.fpsScale))
    );

    return Object.freeze({
        tier,
        mode: effectiveMode,
        inputSize,
        targetFps,
        maxFps,
        minFps: VisionConfig.scheduler.minFps,
        maxDetections: base.maxDetections,
        allowSegmentation: adj.allowSegmentation === null ? base.allowSegmentation : adj.allowSegmentation,
        preferQuantized: base.preferQuantized,
        previewHeight: base.previewHeight
    });
}

/* ------------------------------------------------------------------ browser */

/** True when WASM SIMD is supported. Uses the canonical 8-byte probe module. */
export function detectWasmSimd() {
    try {
        // (module (func (result v128) (v128.const i32x4 0 0 0 0) (drop) ...))
        const probe = new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
            10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
        ]);
        return WebAssembly.validate(probe);
    } catch {
        return false;
    }
}

/**
 * WASM threads need SharedArrayBuffer, which needs cross-origin isolation.
 * GitHub Pages cannot send COOP/COEP, so this is expected to be false on the
 * public deployment. We check rather than assume (requirement 7).
 */
export function detectWasmThreads() {
    const globalScope = typeof self !== "undefined" ? self : globalThis;
    if (typeof SharedArrayBuffer === "undefined") return false;
    if (globalScope.crossOriginIsolated === false) return false;
    return true;
}

/**
 * Probe the runtime environment. Cheap and synchronous except for the optional
 * WebGPU adapter request, which is why it is async.
 * @returns {Promise<object>}
 */
export async function probeDeviceCapabilities() {
    const nav = typeof navigator !== "undefined" ? navigator : {};
    const scr = typeof screen !== "undefined" ? screen : { width: 0, height: 0 };
    const globalScope = typeof self !== "undefined" ? self : globalThis;

    const ua = String(nav.userAgent || "");
    const isIos = /iPad|iPhone|iPod/.test(ua) || (ua.includes("Macintosh") && (nav.maxTouchPoints || 0) > 1);

    let hasWebGpu = false;
    let webGpuAdapterInfo = null;
    if (nav.gpu && typeof nav.gpu.requestAdapter === "function") {
        try {
            const adapter = await nav.gpu.requestAdapter();
            hasWebGpu = Boolean(adapter);
            if (adapter) {
                // `requestAdapterInfo` is not universally available.
                if (adapter.info) {
                    webGpuAdapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture };
                } else if (typeof adapter.requestAdapterInfo === "function") {
                    try {
                        const info = await adapter.requestAdapterInfo();
                        webGpuAdapterInfo = { vendor: info.vendor, architecture: info.architecture };
                    } catch { /* optional */ }
                }
            }
        } catch {
            hasWebGpu = false;
        }
    }

    const dpr = (typeof window !== "undefined" && window.devicePixelRatio) || 1;
    const screenPixels = (scr.width || 0) * (scr.height || 0) * dpr * dpr;

    return {
        hardwareConcurrency: nav.hardwareConcurrency || 4,
        deviceMemoryGb: Number.isFinite(nav.deviceMemory) ? nav.deviceMemory : null,
        hasWebGpu,
        webGpuAdapterInfo,
        hasWasmSimd: detectWasmSimd(),
        hasWasmThreads: detectWasmThreads(),
        crossOriginIsolated: Boolean(globalScope.crossOriginIsolated),
        hasOffscreenCanvas: typeof OffscreenCanvas !== "undefined",
        hasImageBitmap: typeof createImageBitmap === "function",
        hasWorker: typeof Worker !== "undefined",
        hasPerformanceMemory: Boolean(typeof performance !== "undefined" && performance.memory),
        screenPixels,
        devicePixelRatio: dpr,
        isIos,
        userAgentMobile: Boolean(nav.userAgentData?.mobile) || /Android|iPhone|iPad|Mobile/i.test(ua),
        userAgent: ua
    };
}

/**
 * Enumerate camera capabilities without leaving a stream running.
 * Returns null when permission has not been granted yet.
 */
export async function probeCameraCapabilities() {
    const nav = typeof navigator !== "undefined" ? navigator : {};
    if (!nav.mediaDevices?.enumerateDevices) return null;
    try {
        const devices = await nav.mediaDevices.enumerateDevices();
        const cameras = devices.filter((d) => d.kind === "videoinput");
        return {
            count: cameras.length,
            // Labels are empty strings until permission is granted; that is a
            // useful signal in itself.
            labelsVisible: cameras.some((c) => Boolean(c.label)),
            supportedConstraints: nav.mediaDevices.getSupportedConstraints
                ? nav.mediaDevices.getSupportedConstraints()
                : null
        };
    } catch {
        return null;
    }
}

/** Convenience: probe + classify + resolve, in one call. */
export async function buildDeviceProfile(mode = null) {
    const capabilities = await probeDeviceCapabilities();
    const tier = classifyDeviceTier(capabilities);
    const profile = resolveProfile(tier, mode);
    return { capabilities, tier, profile };
}
