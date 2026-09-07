/**
 * Startup robustness.
 *
 * Every test here corresponds to a way the app was observed to hang or fail
 * silently on a real device. A promise that never settles is worse than a
 * rejection: it leaves the UI on "Initialising" with nothing to act on, which is
 * exactly what happened.
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import {
    TimeoutError,
    delay,
    softTimeout,
    withTimeout
} from "../../src/assistive_navigation/web/static/js/utils/async.js";
import {
    QualityMode,
    DeviceTier,
    probeModuleWorker,
    probeOffscreenCanvas2d,
    resolveProfile
} from "../../src/assistive_navigation/web/static/js/config/deviceProfiles.js";
import { MODEL_REGISTRY, getModel } from "../../src/assistive_navigation/web/static/js/config/modelRegistry.js";
import { VisionConfig } from "../../src/assistive_navigation/web/static/js/config/visionConfig.js";
import { CLIENT_ROOT } from "./helpers.mjs";

/* ------------------------------------------------------------- timeouts */

test("withTimeout resolves a fast operation and clears its timer", async () => {
    const value = await withTimeout(Promise.resolve(42), 1000, "fast");
    assert.equal(value, 42);
    // If the timer were left pending, Node's test runner would hang here rather
    // than completing, so reaching this line is the assertion.
});

test("withTimeout rejects a hanging operation with a useful message", async () => {
    const neverSettles = new Promise(() => {});
    await assert.rejects(
        () => withTimeout(neverSettles, 30, "requestAdapter"),
        (error) => {
            assert.ok(error instanceof TimeoutError);
            assert.match(error.message, /requestAdapter timed out after 30 ms/);
            assert.equal(error.timeoutMs, 30);
            return true;
        }
    );
});

test("withTimeout passes through the original rejection", async () => {
    await assert.rejects(
        () => withTimeout(Promise.reject(new Error("session build failed")), 1000, "session"),
        /session build failed/
    );
});

test("withTimeout with a zero budget means unbounded", async () => {
    assert.equal(await withTimeout(Promise.resolve("ok"), 0, "unbounded"), "ok");
});

test("softTimeout turns a hang into a fallback value", async () => {
    // This is the WebGPU probe's contract: not knowing is acceptable, hanging is not.
    const result = await softTimeout(new Promise(() => {}), 20, "unavailable");
    assert.equal(result, "unavailable");
});

test("softTimeout turns a rejection into a fallback value", async () => {
    const result = await softTimeout(Promise.reject(new Error("no adapter")), 500, null);
    assert.equal(result, null);
});

test("softTimeout returns the real value when it arrives in time", async () => {
    const result = await softTimeout(delay(5).then(() => "adapter"), 500, null);
    assert.equal(result, "adapter");
});

/* --------------------------------------------------------- capability probes */

test("capability probes answer without throwing outside a browser", () => {
    // These run during bootstrap, before any error handling is in place, so they
    // must never throw regardless of environment.
    assert.equal(typeof probeOffscreenCanvas2d(), "boolean");
    assert.equal(typeof probeModuleWorker(), "boolean");
});

/* -------------------------------------------------- model / input size match */

/**
 * The exported ONNX graphs have a static input shape. The device profile also
 * carries an input size, and the two can disagree: a LOW-tier device, or FAST
 * mode on any tier, asks for 256 while every ONNX model is 320. Feeding a
 * 256x256 tensor into a 320x320 graph fails every single inference, which
 * presented as "the detector loads but never detects anything".
 *
 * The rule is now: the model dictates the tensor size; the profile only ranks
 * candidates.
 */
test("profile input sizes and model input shapes can legitimately disagree", () => {
    const fastLow = resolveProfile(DeviceTier.LOW, QualityMode.FAST);
    const modelSizes = new Set(MODEL_REGISTRY.map((m) => m.inputSize));

    assert.ok(
        !modelSizes.has(fastLow.inputSize),
        "this test is only meaningful while some profile asks for a size no model provides"
    );
});

test("every ORT model declares an explicit input size for the pipeline to adopt", () => {
    for (const model of MODEL_REGISTRY) {
        if (model.runtime !== "ORT") continue;
        assert.ok(
            VisionConfig.inference.allowedInputSizes.includes(model.inputSize),
            `${model.key} input size ${model.inputSize} is not a supported pipeline size`
        );
    }
});

test("the inference client refuses to resize a static-shape model", async () => {
    const { InferenceClient } = await import(
        "../../src/assistive_navigation/web/static/js/vision/inferenceClient.js"
    );

    const model = getModel("yolo11n-320-uint8");
    const client = new InferenceClient({
        capabilities: { hasWorker: true, hasWasmSimd: true },
        profile: resolveProfile(DeviceTier.MID),
        framePipeline: { inputSize: 320, setInputSize: () => {}, releaseBuffer: () => {} }
    });
    // Pretend a tensor-based adapter is active.
    client.adapter = { needsTensor: true };
    client.model = model;

    const result = await client.setInputSize(256);
    assert.equal(result.ok, false);
    assert.match(result.reason, /static/, result.reason);
});

/* ------------------------------------------- MediaPipe timestamp monotonicity */

/**
 * MediaPipe's VIDEO mode throws on every subsequent call once a timestamp goes
 * backwards. Warm-up used `Date.now()` (~1.7e12) and live inference used
 * `performance.now()` (~1e4), so the first live frame went backwards by a
 * billion milliseconds and the adapter never recovered.
 */
test("the MediaPipe adapter never emits a non-increasing timestamp", async () => {
    const { MediaPipeDetectorAdapter } = await import(
        "../../src/assistive_navigation/web/static/js/vision/adapters/mediapipeAdapter.js"
    );

    const adapter = new MediaPipeDetectorAdapter({
        model: getModel("efficientdet-lite0-int8"),
        modelUrl: "https://example.test/model.tflite",
        detection: { scoreThreshold: 0.3, maxDetections: 16 }
    });

    // A wall-clock warm-up followed by monotonic-clock live frames: the exact
    // sequence that broke.
    const hints = [1.7e12, 1.7e12 + 1, 1.7e12 + 2, 12, 95, 180, 180, 179, 260];
    let previous = -Infinity;

    for (const hint of hints) {
        const value = adapter._nextTimestamp(hint);
        assert.ok(value > previous, `timestamp ${value} did not exceed ${previous} for hint ${hint}`);
        previous = value;
    }
});

test("the MediaPipe adapter tolerates missing timestamp hints", async () => {
    const { MediaPipeDetectorAdapter } = await import(
        "../../src/assistive_navigation/web/static/js/vision/adapters/mediapipeAdapter.js"
    );
    const adapter = new MediaPipeDetectorAdapter({
        model: getModel("efficientdet-lite0-int8"),
        modelUrl: "https://example.test/model.tflite",
        detection: { scoreThreshold: 0.3, maxDetections: 16 }
    });

    let previous = -Infinity;
    for (const hint of [undefined, null, NaN, Infinity, -5]) {
        const value = adapter._nextTimestamp(hint);
        assert.ok(Number.isFinite(value) && value > previous, `bad value ${value} for hint ${hint}`);
        previous = value;
    }
});

/* ------------------------------------------------------------- boot guard */

const BOOT_GUARD = fs.readFileSync(path.join(CLIENT_ROOT, "js/bootGuard.js"), "utf8");
const INDEX_HTML = fs.readFileSync(path.join(CLIENT_ROOT, "index.html"), "utf8");

test("the boot guard is a classic script with no module syntax", () => {
    // It has to run even when the module graph fails to load, which is precisely
    // when it is most needed.
    assert.doesNotMatch(BOOT_GUARD, /^\s*import\s/m, "no import statements");
    assert.doesNotMatch(BOOT_GUARD, /^\s*export\s/m, "no export statements");
});

test("the boot guard avoids syntax that would fail on an old browser", () => {
    // If the browser is too old to parse the guard, the user gets a blank page
    // instead of the message explaining that the browser is too old.
    assert.doesNotMatch(BOOT_GUARD, /=>/, "no arrow functions");
    assert.doesNotMatch(BOOT_GUARD, /\?\./, "no optional chaining");
    assert.doesNotMatch(BOOT_GUARD, /`/, "no template literals");
    assert.doesNotMatch(BOOT_GUARD, /\b(?:const|let)\s/, "no block-scoped declarations");
});

test("the boot guard is loaded before the app module", () => {
    const guardIndex = INDEX_HTML.indexOf("bootGuard.js");
    const appIndex = INDEX_HTML.indexOf("js/app.js");
    assert.ok(guardIndex > -1, "index.html must load the boot guard");
    assert.ok(appIndex > -1, "index.html must load the app");
    assert.ok(guardIndex < appIndex, "the guard must come first so it can capture app load failures");
    assert.doesNotMatch(
        INDEX_HTML.slice(guardIndex - 120, guardIndex),
        /type="module"/,
        "the guard must not be loaded as a module"
    );
});

test("the boot guard captures the failure modes that leave no console", () => {
    for (const hook of ["addEventListener(\"error\"", "unhandledrejection"]) {
        assert.ok(BOOT_GUARD.includes(hook), `missing handler: ${hook}`);
    }
    assert.match(BOOT_GUARD, /isSecureContext/, "must explain an insecure-context camera failure");
});

test("the boot watchdog measures lack of progress, not total elapsed time", () => {
    // A fixed total-time alarm fired mid-startup and told the user the app was
    // stuck while a WebGPU session build was legitimately still running (measured
    // at 24 s under software rendering). Silence is the signal, not duration.
    assert.match(BOOT_GUARD, /NO_PROGRESS_TIMEOUT_MS/, "needs a no-progress budget");
    assert.match(BOOT_GUARD, /lastProgressAt/, "needs to track when progress last happened");
    assert.match(
        BOOT_GUARD,
        /Date\.now\(\) - lastProgressAt < NO_PROGRESS_TIMEOUT_MS/,
        "the watchdog must compare against the last progress, not against startup"
    );
    assert.match(BOOT_GUARD, /setInterval/, "needs to poll rather than fire once");
});

test("the boot guard's DOM targets all exist", () => {
    const ids = [...BOOT_GUARD.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]);
    assert.ok(ids.length >= 3, "expected the guard to bind several elements");
    for (const id of ids) {
        assert.ok(INDEX_HTML.includes(`id="${id}"`), `#${id} is missing from index.html`);
    }
});

test("the app reports each startup step to the trace", async () => {
    const appSource = fs.readFileSync(path.join(CLIENT_ROOT, "js/app.js"), "utf8");
    for (const step of [
        "Device profiled",
        "Pipeline built",
        "Detector ready",
        "Camera started",
        "Camera unavailable",
        "Startup aborted"
    ]) {
        assert.ok(appSource.includes(step), `startup step not traced: ${step}`);
    }
    // The trace helper must be failure-tolerant: diagnostics breaking startup
    // would be a self-inflicted wound.
    assert.match(appSource, /function trace\([\s\S]*?catch/, "trace() must swallow its own errors");
});

test("the whole bootstrap is bounded", async () => {
    const appSource = fs.readFileSync(path.join(CLIENT_ROOT, "js/app.js"), "utf8");
    assert.match(appSource, /BOOTSTRAP_TIMEOUT_MS/, "startup needs an overall ceiling");
    assert.match(
        appSource,
        /withTimeout\(this\._bootstrapSteps\(\), BOOTSTRAP_TIMEOUT_MS/,
        "the ceiling must actually wrap the startup sequence"
    );
});

/* ------------------------------------------------------- runtime sourcing */

test("the worker tries a same-origin runtime before any CDN", () => {
    const worker = fs.readFileSync(path.join(CLIENT_ROOT, "js/workers/inferenceWorker.js"), "utf8");
    const vendorIndex = worker.indexOf("vendor/ort/");
    const cdnIndex = worker.indexOf("unpkg.com");
    assert.ok(vendorIndex > -1, "a self-hosted runtime path must be attempted");
    assert.ok(cdnIndex > -1, "a CDN mirror must exist as a fallback");
    assert.ok(vendorIndex < cdnIndex, "the same-origin copy must be tried first");
});

test("every worker network call is bounded", () => {
    const worker = fs.readFileSync(path.join(CLIENT_ROOT, "js/workers/inferenceWorker.js"), "utf8");
    for (const budget of [
        "ORT_IMPORT_TIMEOUT_MS",
        "MODEL_FETCH_TIMEOUT_MS",
        "SESSION_CREATE_TIMEOUT_MS"
    ]) {
        assert.ok(worker.includes(budget), `missing budget: ${budget}`);
        // Declared and actually applied, not just defined.
        assert.ok(
            new RegExp(`withTimeout\\([\\s\\S]{0,400}?${budget}`).test(worker),
            `${budget} is declared but never applied`
        );
    }
});

test("the frame pipeline verifies its 2D context rather than assuming one", () => {
    const source = fs.readFileSync(path.join(CLIENT_ROOT, "js/vision/framePipeline.js"), "utf8");
    // Safari shipped OffscreenCanvas before 2D context support; assuming a
    // context there means capturing nothing, silently.
    assert.match(source, /createDrawSurface/);
    assert.match(source, /document\.createElement\("canvas"\)/, "needs an element fallback");
    assert.ok(
        !/typeof OffscreenCanvas !== "undefined"\)\s*\{\s*return new OffscreenCanvas/.test(source),
        "must not return an OffscreenCanvas without checking getContext"
    );
});
