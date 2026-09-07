/**
 * Backend selection, device profiling, degradation and recovery.
 *
 * These are the modules that decide what the system does when something is
 * wrong. The recurring theme in the assertions: availability is never treated as
 * proof of usability, and a failure never silently becomes normal operation.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    BackendId,
    buildBackendPlan,
    checkTensorSanity,
    configureOrtEnvironment,
    evaluateWarmup,
    executionProvidersFor,
    percentile
} from "../../src/assistive_navigation/web/static/js/vision/backendSelector.js";
import {
    DeviceTier,
    QualityMode,
    classifyDeviceTier,
    resolveProfile
} from "../../src/assistive_navigation/web/static/js/config/deviceProfiles.js";
import {
    DegradationController,
    DegradationLevel
} from "../../src/assistive_navigation/web/static/js/health/degradation.js";
import {
    RecoveryManager,
    RecoveryStep,
    VisionWatchdog
} from "../../src/assistive_navigation/web/static/js/health/recoveryManager.js";
import { VisionHealthMonitor, HealthState } from "../../src/assistive_navigation/web/static/js/health/visionHealthMonitor.js";
import {
    MODEL_REGISTRY,
    rankCandidates,
    resolveModelUrl,
    getModel
} from "../../src/assistive_navigation/web/static/js/config/modelRegistry.js";
import { VisionConfig } from "../../src/assistive_navigation/web/static/js/config/visionConfig.js";
import { Clock } from "./helpers.mjs";

/* -------------------------------------------------------- backend plan */

test("WebGPU is attempted first when present, but is never the only option", () => {
    const plan = buildBackendPlan({ hasWebGpu: true, hasWasmSimd: true, hasWorker: true });
    assert.equal(plan[0].id, BackendId.WEBGPU);
    assert.ok(plan.length > 1, "there is always something to fall back to");
    assert.ok(plan.some((p) => p.id === BackendId.WASM_SIMD));
    assert.ok(plan.some((p) => p.id === BackendId.WASM));
});

test("a device without WebGPU still gets a complete plan", () => {
    const plan = buildBackendPlan({ hasWebGpu: false, hasWasmSimd: true, hasWorker: true });
    assert.ok(!plan.some((p) => p.id === BackendId.WEBGPU));
    assert.equal(plan[0].id, BackendId.WASM_SIMD);
});

test("a device without SIMD still gets baseline WASM", () => {
    const plan = buildBackendPlan({ hasWebGpu: false, hasWasmSimd: false, hasWorker: true });
    assert.ok(!plan.some((p) => p.id === BackendId.WASM_SIMD));
    assert.equal(plan[0].id, BackendId.WASM);
});

test("a device without workers falls back to main-thread runtimes only", () => {
    const plan = buildBackendPlan({ hasWorker: false, hasWasmSimd: true });
    assert.ok(plan.every((p) => p.runtime !== "ORT"), "ORT requires the worker");
    assert.ok(plan.some((p) => p.runtime === "MEDIAPIPE"));
    assert.ok(plan.some((p) => p.runtime === "TFJS"));
});

test("the plan records why each backend was chosen, including thread availability", () => {
    const isolated = buildBackendPlan({ hasWasmSimd: true, hasWasmThreads: true, hasWorker: true });
    const notIsolated = buildBackendPlan({ hasWasmSimd: true, hasWasmThreads: false, hasWorker: true });
    const simdIsolated = isolated.find((p) => p.id === BackendId.WASM_SIMD);
    const simdPlain = notIsolated.find((p) => p.id === BackendId.WASM_SIMD);

    assert.ok(simdIsolated.reason.includes("threads"));
    assert.ok(simdPlain.reason.includes("single thread"));
});

/* ------------------------------------------------------- warm-up gate */

test("percentile handles empty and single-element inputs", () => {
    assert.equal(percentile([], 95), 0);
    assert.equal(percentile([42], 95), 42);
    assert.equal(percentile([10, 20, 30, 40], 50), 20);
});

test("a backend that produces nothing is rejected", () => {
    const verdict = evaluateWarmup({ latencies: [10, 12], producedOutput: false, outputSane: false }, 700);
    assert.equal(verdict.accepted, false);
    assert.ok(verdict.reason.includes("no output"));
});

test("a backend that produces garbage is rejected even if it is fast", () => {
    const verdict = evaluateWarmup({ latencies: [5, 6, 5], producedOutput: true, outputSane: false }, 700);
    assert.equal(verdict.accepted, false);
    assert.ok(verdict.reason.includes("sanity"));
});

test("a backend with an unacceptable p95 is rejected even if its mean is fine", () => {
    // Warm-up runs a handful of passes, so with 8 samples the 95th percentile is
    // the slowest one. Mean is ~145 ms, which looks acceptable; the tail is not.
    const latencies = [40, 45, 42, 41, 44, 43, 46, 900];
    const verdict = evaluateWarmup({ latencies, producedOutput: true, outputSane: true }, 700);
    assert.equal(verdict.accepted, false, `p95 was ${verdict.p95}`);
    assert.ok(verdict.reason.includes("p95"), verdict.reason);
    assert.ok(verdict.mean < 200, "the mean alone would have let this through");
});

test("percentile indexing is nearest-rank, so a lone outlier in 20 samples is p100", () => {
    // Documenting the semantics rather than assuming them: with 20 samples the
    // 95th percentile is the 19th value, so a single spike does not move it.
    // This is why the warm-up gate uses a small number of passes - it makes the
    // gate strict about tails rather than forgiving of them.
    const latencies = Array.from({ length: 19 }, () => 42).concat([900]);
    assert.equal(percentile(latencies, 95), 42);
    assert.equal(percentile(latencies, 100), 900);
    assert.ok(VisionConfig.warmup.passes <= 8, "warm-up keeps its sample count small on purpose");
});

test("a healthy backend is accepted", () => {
    const verdict = evaluateWarmup(
        { latencies: [80, 55, 52, 54], producedOutput: true, outputSane: true },
        700
    );
    assert.equal(verdict.accepted, true);
    assert.equal(verdict.reason, "passed");
});

test("tensor sanity rejects empty, NaN and all-zero output", () => {
    assert.equal(checkTensorSanity(new Float32Array(0)).sane, false);
    assert.equal(checkTensorSanity(new Float32Array([1, 2, NaN, 4])).sane, false);
    assert.equal(checkTensorSanity(new Float32Array(2048)).sane, false, "an all-zero head is not plausible");

    const plausible = new Float32Array(2048);
    for (let i = 0; i < plausible.length; i += 1) plausible[i] = (i % 97) / 97;
    assert.equal(checkTensorSanity(plausible).sane, true);
});

test("execution providers keep WASM as an in-session fallback under WebGPU", () => {
    assert.deepEqual(executionProvidersFor(BackendId.WEBGPU), ["webgpu", "wasm"]);
    assert.deepEqual(executionProvidersFor(BackendId.WASM_SIMD), ["wasm"]);
});

test("threads are only requested when SharedArrayBuffer is genuinely usable", () => {
    const ort = { env: { wasm: {} } };

    const isolated = configureOrtEnvironment(ort, BackendId.WASM_SIMD, {
        hasWasmThreads: true,
        hardwareConcurrency: 8
    });
    assert.ok(isolated.numThreads > 1);
    assert.ok(isolated.numThreads <= 4, "never oversubscribe");
    assert.equal(isolated.simd, true);

    const notIsolated = configureOrtEnvironment(ort, BackendId.WASM_SIMD, {
        hasWasmThreads: false,
        hardwareConcurrency: 8
    });
    assert.equal(notIsolated.numThreads, 1, "single thread without cross-origin isolation");

    const baseline = configureOrtEnvironment(ort, BackendId.WASM, {
        hasWasmThreads: true,
        hardwareConcurrency: 8
    });
    assert.equal(baseline.simd, false);
    assert.equal(baseline.numThreads, 1);
});

/* ----------------------------------------------------- device profiles */

test("device tiers reflect reported capability", () => {
    assert.equal(
        classifyDeviceTier({ hardwareConcurrency: 8, deviceMemoryGb: 8, hasWebGpu: true }),
        DeviceTier.HIGH
    );
    assert.equal(
        classifyDeviceTier({ hardwareConcurrency: 2, deviceMemoryGb: 2, hasWebGpu: false }),
        DeviceTier.LOW
    );
    assert.equal(
        classifyDeviceTier({ hardwareConcurrency: 6, deviceMemoryGb: 4, hasWebGpu: false }),
        DeviceTier.MID
    );
});

test("an uninformative device lands in MID, not HIGH", () => {
    // Guessing high and thermally collapsing mid-demo is the worse failure.
    assert.equal(classifyDeviceTier({}), DeviceTier.MID);
});

test("profiles stay inside the scheduler's configured bounds", () => {
    for (const tier of Object.values(DeviceTier)) {
        for (const mode of Object.values(QualityMode)) {
            const profile = resolveProfile(tier, mode);
            assert.ok(profile.targetFps >= VisionConfig.scheduler.minFps, `${tier}/${mode} target too low`);
            assert.ok(profile.maxFps <= VisionConfig.scheduler.maxFps, `${tier}/${mode} max too high`);
            assert.ok(profile.maxFps >= profile.targetFps);
            assert.ok(VisionConfig.inference.allowedInputSizes.includes(profile.inputSize));
        }
    }
});

test("accuracy mode raises input size and lowers the rate target", () => {
    const balanced = resolveProfile(DeviceTier.HIGH, QualityMode.BALANCED);
    const accuracy = resolveProfile(DeviceTier.HIGH, QualityMode.ACCURACY);
    const fast = resolveProfile(DeviceTier.HIGH, QualityMode.FAST);

    assert.ok(accuracy.inputSize >= balanced.inputSize);
    assert.ok(accuracy.targetFps < balanced.targetFps, "accuracy costs rate, and admits it");
    assert.ok(fast.inputSize <= balanced.inputSize);
    assert.ok(fast.targetFps >= balanced.targetFps);
    assert.equal(fast.allowSegmentation, false, "fast mode drops the optional stage");
});

test("low-end devices get a smaller model, smaller input and no refinement", () => {
    const low = resolveProfile(DeviceTier.LOW);
    assert.equal(low.allowSegmentation, false);
    assert.equal(low.preferQuantized, true);
    assert.ok(low.inputSize <= 320);
    assert.ok(low.targetFps <= 10);
});

/* ----------------------------------------------------- model registry */

test("every registry entry declares the fields the selector relies on", () => {
    for (const model of MODEL_REGISTRY) {
        assert.ok(model.key, "key");
        assert.ok(model.label, `${model.key} label`);
        assert.ok(model.runtime, `${model.key} runtime`);
        assert.ok(model.decode, `${model.key} decode family`);
        assert.ok(Array.isArray(model.supportedBackends) && model.supportedBackends.length, `${model.key} backends`);
        assert.ok(model.inputSize > 0, `${model.key} inputSize`);
        assert.ok(model.quantization, `${model.key} quantization`);
        assert.ok(model.assetPath || model.assetUrl || model.isLastResort, `${model.key} has no asset`);
    }
});

test("quantised candidates are preferred on the WASM path", () => {
    const ranked = rankCandidates({ backendId: BackendId.WASM_SIMD, preferQuantized: true });
    assert.ok(ranked.length > 1);
    assert.notEqual(ranked[0].quantization, "fp32");
});

test("fp32 is preferred on the WebGPU path", () => {
    const ranked = rankCandidates({ backendId: BackendId.WEBGPU, preferQuantized: false });
    assert.ok(ranked.length >= 1);
    assert.equal(ranked[0].quantization, "fp32");
});

test("the last-resort detector always sorts last", () => {
    const ranked = rankCandidates({ backendId: "tfjs-webgl", preferQuantized: true });
    assert.equal(ranked[ranked.length - 1].isLastResort, true);
});

test("same-origin model paths resolve against the page URL", () => {
    const model = getModel("yolo11n-320-uint8");
    const url = resolveModelUrl(model, "https://example.test/app/");
    assert.equal(url, "https://example.test/app/models/yolo11n_320_uint8.onnx");
});

/* -------------------------------------------------------- degradation */

test("degradation is one-way unless a reason is given to restore", () => {
    const controller = new DegradationController();
    assert.equal(controller.level, DegradationLevel.FULL);

    assert.equal(controller.degradeTo(DegradationLevel.NO_REFINEMENT, "stage too slow"), true);
    assert.equal(controller.degradeTo(DegradationLevel.FULL, "looks fine now"), false, "degradeTo never improves");
    assert.equal(controller.level, DegradationLevel.NO_REFINEMENT);

    assert.equal(controller.restoreTo(DegradationLevel.FULL, "measured headroom"), true);
    assert.equal(controller.level, DegradationLevel.FULL);
});

test("level 3 stops the system offering directions at all", () => {
    const controller = new DegradationController();
    controller.degradeTo(DegradationLevel.DETECTOR_ONLY, "tracking unusable");
    assert.equal(controller.capabilities.directionalGuidance, false);
    assert.equal(controller.capabilities.approachDetection, false);
    assert.equal(controller.capabilities.detector, true, "it still detects");
});

test("evaluate maps conditions onto the right level", () => {
    const healthy = DegradationController.evaluate({
        detectorWorking: true,
        refinementEnabled: true,
        inferenceFps: 12,
        trackingStability: 0.8,
        perceptionUsable: true
    });
    assert.equal(healthy.level, DegradationLevel.FULL);

    const noRefinement = DegradationController.evaluate({
        detectorWorking: true,
        refinementEnabled: false,
        inferenceFps: 12,
        trackingStability: 0.8,
        perceptionUsable: true
    });
    assert.equal(noRefinement.level, DegradationLevel.NO_REFINEMENT);

    const tooSlow = DegradationController.evaluate({
        detectorWorking: true,
        refinementEnabled: true,
        inferenceFps: 2.5,
        trackingStability: 0.8,
        perceptionUsable: true
    });
    assert.equal(tooSlow.level, DegradationLevel.DETECTOR_ONLY);
    assert.ok(tooSlow.reason.includes("too low"));

    const blind = DegradationController.evaluate({
        detectorWorking: false,
        refinementEnabled: false,
        inferenceFps: 0,
        trackingStability: null,
        perceptionUsable: true
    });
    assert.equal(blind.level, DegradationLevel.UNRELIABLE);
});

test("degrading is immediate but restoring needs repeated confirmation", () => {
    const controller = new DegradationController();
    const bad = { level: DegradationLevel.DETECTOR_ONLY, reason: "slow" };
    const good = { level: DegradationLevel.FULL, reason: "recovered" };

    assert.equal(controller.apply(bad), true, "degrade at once");
    assert.equal(controller.apply(good), false, "one good sample is not enough");
    for (let i = 0; i < 3; i += 1) controller.apply(good);
    assert.equal(controller.level, DegradationLevel.DETECTOR_ONLY, "still not enough");
    controller.apply(good);
    assert.equal(controller.level, DegradationLevel.FULL, "restored after sustained evidence");
});

/* ---------------------------------------------------------- watchdog */

test("the watchdog fires only after the configured silence", async () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });
    let stalls = 0;

    const watchdog = new VisionWatchdog({
        monitor,
        onStall: () => { stalls += 1; },
        intervalMs: 10,
        now: clock.now
    });

    monitor.markInferenceSuccess();
    watchdog.start();

    clock.advance(VisionConfig.health.watchdogTimeoutMs - 100);
    watchdog._tick();
    assert.equal(stalls, 0, "inside the tolerance window");

    clock.advance(200);
    watchdog._tick();
    assert.equal(stalls, 1, "past the tolerance window");

    // It must not re-fire immediately while recovery is running.
    watchdog._tick();
    assert.equal(stalls, 1);

    watchdog.stop();
});

test("health state tracks silence and consecutive failures", () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });

    monitor.markInferenceSuccess();
    assert.equal(monitor.state, HealthState.HEALTHY);
    assert.equal(monitor.isStalled, false);

    clock.advance(VisionConfig.health.watchdogTimeoutMs + 100);
    assert.equal(monitor.isStalled, true, "silence is detected, not ignored");

    for (let i = 0; i < VisionConfig.health.maxConsecutiveFailures; i += 1) {
        monitor.markInferenceFailure("boom");
    }
    assert.equal(monitor.state, HealthState.FAILING);

    monitor.markInferenceSuccess();
    assert.equal(monitor.state, HealthState.HEALTHY);
    assert.equal(monitor.consecutiveInferenceFailures, 0);
});

test("heap growth is reported rather than assumed absent", () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });

    // No performance.memory: reported as unavailable, not as zero growth.
    assert.equal(monitor.sampleMemory({ memory: undefined }), null);
    assert.equal(monitor.snapshot().heapUsedMb, null);

    let used = 50e6;
    const fakePerf = { get memory() { return { usedJSHeapSize: used, jsHeapSizeLimit: 2e9 }; } };
    clock.advance(VisionConfig.health.heapSampleIntervalMs + 1);
    monitor.sampleMemory(fakePerf);

    used = 50e6 + VisionConfig.health.heapGrowthWarningBytes + 10e6;
    clock.advance(VisionConfig.health.heapSampleIntervalMs + 1);
    monitor.sampleMemory(fakePerf);

    assert.equal(monitor.memoryWarnings, 1);
});

/* ---------------------------------------------------------- recovery */

test("the recovery ladder is climbed in order and reported", async () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });
    const attempted = [];

    const handlers = {};
    for (const step of [
        RecoveryStep.RETRY,
        RecoveryStep.RESTART_WORKER,
        RecoveryStep.FALLBACK_BACKEND,
        RecoveryStep.RELOAD_MODEL,
        RecoveryStep.REINITIALISE
    ]) {
        handlers[step] = async () => {
            attempted.push(step);
            return { ok: false, detail: "still broken" };
        };
    }

    let exhaustedCalls = 0;
    const manager = new RecoveryManager({
        monitor,
        handlers,
        cfg: { ...VisionConfig.health, recoveryBackoffMs: [0, 0, 0, 0], maxRecoveryAttempts: 5 },
        onExhausted: () => { exhaustedCalls += 1; }
    });

    for (let i = 0; i < 6; i += 1) await manager.escalate();

    assert.deepEqual(attempted, [
        RecoveryStep.RETRY,
        RecoveryStep.RESTART_WORKER,
        RecoveryStep.FALLBACK_BACKEND,
        RecoveryStep.RELOAD_MODEL,
        RecoveryStep.REINITIALISE
    ]);
    assert.equal(manager.exhausted, true);
    assert.ok(exhaustedCalls >= 1, "exhaustion is announced, not swallowed");
    assert.equal(monitor.state, HealthState.DEAD);
});

test("a cheap recovery is retried first after the pipeline recovers", async () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });
    const attempted = [];
    const handlers = {
        [RecoveryStep.RETRY]: async () => {
            attempted.push(RecoveryStep.RETRY);
            return { ok: true };
        },
        [RecoveryStep.RESTART_WORKER]: async () => {
            attempted.push(RecoveryStep.RESTART_WORKER);
            return { ok: true };
        }
    };
    const manager = new RecoveryManager({
        monitor,
        handlers,
        cfg: { ...VisionConfig.health, recoveryBackoffMs: [0, 0, 0, 0] }
    });

    const first = await manager.escalate();
    assert.equal(first.step, RecoveryStep.RETRY);
    assert.equal(first.ok, true);

    // A successful step does not by itself mean the pipeline is producing
    // results, so the ladder only resets once inference actually succeeds.
    manager.notifySuccess();
    const second = await manager.escalate();
    assert.equal(second.step, RecoveryStep.RETRY, "back to the cheapest step for an unrelated failure");
    assert.deepEqual(attempted, [RecoveryStep.RETRY, RecoveryStep.RETRY]);
});

test("a throwing handler is treated as a failed step, not a crash", async () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });
    const manager = new RecoveryManager({
        monitor,
        handlers: {
            [RecoveryStep.RETRY]: async () => { throw new Error("worker gone"); }
        },
        cfg: { ...VisionConfig.health, recoveryBackoffMs: [0, 0, 0, 0] }
    });

    const result = await manager.escalate();
    assert.equal(result.ok, false);
    assert.ok(String(result.detail).includes("worker gone"));
    assert.equal(manager.metrics().nextStep, RecoveryStep.RESTART_WORKER);
});

test("concurrent escalations are refused", async () => {
    const clock = new Clock(0);
    const monitor = new VisionHealthMonitor({ now: clock.now });
    let release;
    const manager = new RecoveryManager({
        monitor,
        handlers: {
            [RecoveryStep.RETRY]: () => new Promise((r) => { release = () => r({ ok: true }); })
        },
        cfg: { ...VisionConfig.health, recoveryBackoffMs: [0, 0, 0, 0] }
    });

    const first = manager.escalate();
    await new Promise((r) => setTimeout(r, 5));
    const second = await manager.escalate();
    assert.equal(second.ok, false);
    assert.equal(second.detail, "already recovering");

    release();
    await first;
});
