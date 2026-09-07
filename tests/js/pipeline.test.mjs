/**
 * Scheduling, frame quality, scene change and the refinement stage.
 *
 * The scheduler tests encode the single most important runtime rule in this
 * system: **never queue a frame**. Everything else here is about tolerating the
 * mess that walking makes of a camera feed without either panicking or lying.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    DropReason,
    FrameScheduler,
    LatencyWindow
} from "../../src/assistive_navigation/web/static/js/vision/frameScheduler.js";
import {
    FrameQualityMonitor,
    QualityState,
    analyseLuminance,
    scoreFrame
} from "../../src/assistive_navigation/web/static/js/vision/frameQuality.js";
import {
    SceneChangeDetector,
    SceneEvent,
    columnProfile,
    estimateHorizontalShift,
    meanAbsoluteDifference
} from "../../src/assistive_navigation/web/static/js/vision/sceneChange.js";
import {
    AdaptiveSegmentationController,
    GroundFreenessEstimator,
    SegmentationTrigger
} from "../../src/assistive_navigation/web/static/js/perception/segmentationStage.js";
import { VisionConfig } from "../../src/assistive_navigation/web/static/js/config/visionConfig.js";
import { lumaPlane } from "./helpers.mjs";

/* ------------------------------------------------------ latency window */

test("LatencyWindow computes percentiles over a bounded ring", () => {
    const w = new LatencyWindow(5);
    for (const v of [10, 20, 30, 40, 50, 60]) w.push(v);
    assert.equal(w.length, 5, "capacity is respected");
    assert.equal(w.percentile(50), 40);
    assert.equal(w.percentile(100), 60);
    assert.equal(w.max(), 60);
    assert.equal(w.mean(), (20 + 30 + 40 + 50 + 60) / 5);
});

test("LatencyWindow ignores non-finite samples", () => {
    const w = new LatencyWindow(4);
    w.push(10);
    w.push(NaN);
    w.push(Infinity);
    assert.equal(w.length, 1);
});

/* ---------------------------------------------------------- scheduling */

test("no frame is submitted while inference is in flight", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 12 });
    let now = 1000;

    const first = scheduler.shouldRun(now, { busy: false, canCapture: true });
    assert.equal(first.run, true);
    scheduler.markSubmitted(now);

    // Many render frames pass while inference is busy. Every one is refused.
    for (let i = 0; i < 20; i += 1) {
        now += 16;
        const gate = scheduler.shouldRun(now, { busy: true, canCapture: true });
        assert.equal(gate.run, false);
        assert.equal(gate.reason, DropReason.BUSY);
        scheduler.markDropped(gate.reason);
    }

    assert.equal(scheduler.counters.submitted, 1, "exactly one submission, no queue");
    assert.equal(scheduler.counters.dropped, 20);
});

test("the newest frame is used once inference frees up", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 10 });
    let now = 1000;
    scheduler.markSubmitted(now);
    scheduler.markCompleted(now + 50, 50, 55);

    now += 200; // past the 100 ms interval
    const gate = scheduler.shouldRun(now, { busy: false, canCapture: true });
    assert.equal(gate.run, true, "the next decision is made against a fresh frame");
});

test("rate limiting is not counted as a dropped frame", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 10 });
    let now = 1000;
    scheduler.markSubmitted(now);

    for (let i = 0; i < 5; i += 1) {
        now += 10;
        const gate = scheduler.shouldRun(now, { busy: false, canCapture: true });
        assert.equal(gate.reason, DropReason.RATE_LIMIT);
        scheduler.markDropped(gate.reason);
    }
    assert.equal(scheduler.counters.dropped, 0, "running at 10 FPS on a 60 FPS loop is not a fault");
    assert.equal(scheduler.counters.byReason[DropReason.RATE_LIMIT], 5, "but it is still observable");
});

test("an exhausted buffer pool refuses capture", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 12 });
    const gate = scheduler.shouldRun(5000, { busy: false, canCapture: false });
    assert.equal(gate.run, false);
    assert.equal(gate.reason, DropReason.NO_BUFFER);
});

test("the controller sheds rate when p95 latency is too high", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 15, maxFps: 20 });
    let now = 1000;
    const before = scheduler.targetFps;

    for (let i = 0; i < 20; i += 1) {
        now += 60;
        scheduler.markCompleted(now, VisionConfig.scheduler.maxP95LatencyMs + 80);
    }
    now += VisionConfig.scheduler.controlIntervalMs + 10;
    const result = scheduler.control(now);

    assert.ok(result, "the controller ran");
    assert.ok(scheduler.targetFps < before, `${scheduler.targetFps} should be below ${before}`);
    assert.equal(result.action, "shed-latency");
});

test("the controller ramps up when there is headroom", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 8, maxFps: 20 });
    let now = 1000;

    for (let window = 0; window < VisionConfig.scheduler.rampUpWindows + 1; window += 1) {
        for (let i = 0; i < 12; i += 1) {
            now += 40;
            scheduler.markCompleted(now, 25);
        }
        now += VisionConfig.scheduler.controlIntervalMs + 10;
        scheduler.control(now);
    }
    assert.ok(scheduler.targetFps > 8, `never ramped up, still ${scheduler.targetFps}`);
});

test("sustained degradation is treated as thermal throttling", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 15, maxFps: 20 });
    let now = 1000;

    // Establish a fast baseline.
    for (let i = 0; i < 20; i += 1) {
        now += 50;
        scheduler.markCompleted(now, 30);
    }
    now += VisionConfig.scheduler.controlIntervalMs + 10;
    scheduler.control(now);
    assert.equal(scheduler.bestP95, 30);

    // Now everything gets slower, though still nominally inside the absolute
    // budget. Relative regression is what exposes throttling.
    scheduler.latency.clear();
    for (let i = 0; i < 20; i += 1) {
        now += 80;
        scheduler.markCompleted(now, 70);
    }
    now += VisionConfig.scheduler.controlIntervalMs + 10;
    const result = scheduler.control(now);

    assert.equal(result.action, "shed-thermal");
    assert.equal(scheduler.thermalThrottled, true);
    assert.ok(scheduler.ceilingFps < 20, "the ceiling drops so ramp-up cannot undo it");
});

test("the target rate never leaves its configured bounds", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 12, maxFps: 18, minFps: 5 });
    let now = 1000;
    for (let round = 0; round < 30; round += 1) {
        for (let i = 0; i < 12; i += 1) {
            now += 40;
            scheduler.markCompleted(now, 900);
        }
        now += VisionConfig.scheduler.controlIntervalMs + 10;
        scheduler.control(now);
        assert.ok(scheduler.targetFps >= 5, `dropped to ${scheduler.targetFps}`);
        assert.ok(scheduler.targetFps <= 18, `rose to ${scheduler.targetFps}`);
    }
});

test("scheduler metrics separate camera and inference rates", () => {
    const scheduler = new FrameScheduler(VisionConfig.scheduler, { targetFps: 10 });
    let now = 1000;
    for (let i = 0; i < 60; i += 1) {
        now += 16.7;
        scheduler.tickCamera(now);
        if (i % 6 === 0) scheduler.markCompleted(now, 40);
    }
    const m = scheduler.metrics();
    assert.ok(m.cameraFps > 50, `camera ${m.cameraFps}`);
    assert.ok(m.inferenceFps > 5 && m.inferenceFps < 15, `inference ${m.inferenceFps}`);
});

/* ------------------------------------------------------- frame quality */

test("analyseLuminance measures brightness, contrast and gradient energy", () => {
    const flat = lumaPlane(64, 48, () => 128);
    const flatStats = analyseLuminance(flat, 64, 48);
    assert.equal(flatStats.brightness, 128);
    assert.equal(flatStats.contrast, 0);
    assert.equal(flatStats.blur, 0, "a flat field has no gradient energy");

    const checker = lumaPlane(64, 48, (x, y) => ((x + y) % 2 ? 220 : 20));
    const checkerStats = analyseLuminance(checker, 64, 48);
    assert.ok(checkerStats.blur > flatStats.blur);
    assert.ok(checkerStats.contrast > 90);
});

test("scoreFrame flags dark, blurred and low-contrast frames", () => {
    const dark = scoreFrame({ brightness: 10, contrast: 30, blur: 0.007 });
    assert.ok(dark.issues.includes("dark"));
    assert.ok(dark.score < 0.7);

    // 0.0005 is roughly what a heavily smeared thumbnail measures.
    const blurred = scoreFrame({ brightness: 128, contrast: 30, blur: 0.0005 });
    assert.ok(blurred.issues.includes("blur"));

    const featureless = scoreFrame({ brightness: 128, contrast: 2, blur: 0.007 });
    assert.ok(featureless.issues.includes("low-contrast"));

    // 0.007 is roughly what a structured indoor scene measures.
    const good = scoreFrame({ brightness: 128, contrast: 40, blur: 0.007 });
    assert.equal(good.issues.length, 0, `unexpected issues: ${good.issues.join(",")}`);
    assert.ok(good.score > 0.85, `score ${good.score}`);
});

test("the blur measure separates a smeared frame from a structured one", () => {
    // Both are mid-grey with the same mean; only the structure differs.
    const structured = lumaPlane(64, 48, (x, y) => {
        let v = 150 - y * 1.2;
        if (x > 10 && x < 24 && y > 18 && y < 40) v = 55;
        if (x > 38 && x < 56 && y > 10 && y < 30) v = 95;
        return v;
    });
    const smeared = lumaPlane(64, 48, (x) => 120 + 8 * Math.sin(x / 9));

    const structuredBlur = analyseLuminance(structured, 64, 48).blur;
    const smearedBlur = analyseLuminance(smeared, 64, 48).blur;

    assert.ok(structuredBlur > VisionConfig.quality.blurThreshold, `structured ${structuredBlur}`);
    assert.ok(smearedBlur < VisionConfig.quality.blurThreshold, `smeared ${smearedBlur}`);
    assert.ok(structuredBlur > smearedBlur * 5, "with a comfortable margin between them");
});

/** A plausible indoor scene: a lighting gradient plus a few hard object edges. */
function structuredScene() {
    return lumaPlane(64, 48, (x, y) => {
        let v = 150 - y * 1.2;
        if (x > 10 && x < 24 && y > 18 && y < 40) v = 55;
        if (x > 38 && x < 56 && y > 10 && y < 30) v = 95;
        if (y > 40) v = 110 + ((x % 6 < 3) ? 12 : -12);
        return v;
    });
}

test("one or two bad frames do not degrade the system", () => {
    const monitor = new FrameQualityMonitor();
    const good = structuredScene();
    const black = lumaPlane(64, 48, () => 3);

    for (let i = 0; i < 6; i += 1) monitor.update(good, 64, 48);
    assert.equal(monitor.state, QualityState.GOOD, `rolling score ${monitor.rollingScore}`);

    monitor.update(black, 64, 48);
    monitor.update(black, 64, 48);
    assert.equal(monitor.state, QualityState.GOOD, "a footfall or a passing shadow is not a failure");
});

test("sustained poor frames degrade, and recovery needs sustained good frames", () => {
    const monitor = new FrameQualityMonitor();
    const good = structuredScene();
    const black = lumaPlane(64, 48, () => 3);

    for (let i = 0; i < 6; i += 1) monitor.update(good, 64, 48);
    for (let i = 0; i < VisionConfig.quality.degradeAfterPoorFrames + 2; i += 1) monitor.update(black, 64, 48);
    assert.notEqual(monitor.state, QualityState.GOOD);
    assert.equal(monitor.dominantIssue, "dark");

    monitor.update(good, 64, 48);
    assert.notEqual(monitor.state, QualityState.GOOD, "one good frame is not recovery");
    for (let i = 0; i < VisionConfig.quality.recoverAfterGoodFrames + 1; i += 1) monitor.update(good, 64, 48);
    assert.equal(monitor.state, QualityState.GOOD);
});

test("an unusable feed withholds directions and asks for a stop", () => {
    const monitor = new FrameQualityMonitor();
    const black = lumaPlane(64, 48, () => 1);
    for (let i = 0; i < 16; i += 1) monitor.update(black, 64, 48);
    assert.equal(monitor.state, QualityState.UNUSABLE);
    assert.equal(monitor.requiresStop, true);
    assert.equal(monitor.allowsDirectionalGuidance, false);
});

/* -------------------------------------------------------- scene change */

test("meanAbsoluteDifference is 0 for identical planes and 1 for inverted", () => {
    const a = lumaPlane(8, 8, () => 0);
    const b = lumaPlane(8, 8, () => 255);
    assert.equal(meanAbsoluteDifference(a, a, 64), 0);
    assert.equal(meanAbsoluteDifference(a, b, 64), 1);
});

test("columnProfile averages down the vertical axis", () => {
    const luma = lumaPlane(4, 3, (x) => x * 60);
    const out = new Float32Array(4);
    columnProfile(luma, 4, 3, out);
    assert.deepEqual(Array.from(out), [0, 60, 120, 180]);
});

test("estimateHorizontalShift recovers a known pan", () => {
    const width = 32;
    const prev = new Float32Array(width);
    for (let x = 0; x < width; x += 1) prev[x] = 40 + 6 * ((x * 7) % 11);

    const shift = 4;
    const curr = new Float32Array(width);
    for (let x = 0; x < width; x += 1) curr[x] = prev[(x - shift + width) % width];

    const result = estimateHorizontalShift(prev, curr, width, 8);
    assert.equal(result.shiftColumns, shift);
    assert.ok(result.confidence > 0.3, `weak confidence ${result.confidence}`);
});

test("a static scene reports STABLE and tracks stay trusted", () => {
    const detector = new SceneChangeDetector();
    const luma = lumaPlane(32, 24, (x, y) => 100 + 40 * Math.sin(x / 2) + 10 * y);

    detector.update(luma, 32, 24);
    const result = detector.update(luma, 32, 24);
    assert.equal(result.event, SceneEvent.STABLE);
    assert.equal(result.trustTracks, true);
});

test("a large pan is reported as a turn and revokes track trust", () => {
    const detector = new SceneChangeDetector();
    const pattern = (offset) => lumaPlane(32, 24, (x, y) => 80 + 70 * Math.sin((x + offset) / 1.7) + 6 * (y % 4));

    detector.update(pattern(0), 32, 24);
    const result = detector.update(pattern(7), 32, 24);

    assert.ok(
        result.event === SceneEvent.TURN || result.event === SceneEvent.VIOLENT_MOTION,
        `expected a turn, got ${result.event}`
    );
    assert.equal(result.trustTracks, false, "tracks from before a turn are not trustworthy");
});

test("a content change without a pan is a scene change, and tracks survive", () => {
    const detector = new SceneChangeDetector();
    // Random-looking but deterministic textures with no horizontal relationship.
    const first = lumaPlane(32, 24, (x, y) => (x * 37 + y * 11) % 256);
    const second = lumaPlane(32, 24, (x, y) => (x * 53 + y * 97 + 128) % 256);

    detector.update(first, 32, 24);
    const result = detector.update(second, 32, 24);
    assert.notEqual(result.event, SceneEvent.STABLE);
    if (result.event === SceneEvent.SCENE_CHANGE) {
        assert.equal(result.trustTracks, true, "geometry is still roughly valid");
    }
});

/* ---------------------------------------------------- refinement stage */

test("ground freeness finds the row where the floor ends", () => {
    const width = 64;
    const height = 48;
    // Uniform floor below row 30, a distinctly different obstacle above it in
    // the middle columns.
    const luma = lumaPlane(width, height, (x, y) => {
        if (y >= 30) return 120;
        if (x >= 24 && x < 40) return 30;
        return 120;
    });

    const estimator = new GroundFreenessEstimator({ columns: 16 });
    const { columns } = estimator.estimate(luma, width, height);

    const blockedColumn = columns[7];   // maps into x 28..32
    const openColumn = columns[1];      // maps into x 4..8
    assert.ok(blockedColumn < openColumn, `blocked ${blockedColumn} vs open ${openColumn}`);
    assert.ok(openColumn > 0.9, `an unobstructed column should read open, got ${openColumn}`);
});

test("ground freeness reports an entirely open floor as open", () => {
    const luma = lumaPlane(64, 48, () => 118);
    const estimator = new GroundFreenessEstimator({ columns: 16 });
    const { columns } = estimator.estimate(luma, 64, 48);
    for (let i = 0; i < columns.length; i += 1) {
        assert.equal(columns[i], 1, `column ${i} read as blocked on a plain floor`);
    }
});

test("the refinement governor runs on triggers and skips otherwise", () => {
    const estimator = new GroundFreenessEstimator({ columns: 16 });
    const controller = new AdaptiveSegmentationController({ estimator, periodicIntervalMs: 400 });
    const luma = lumaPlane(64, 48, () => 118);
    const base = { luma, width: 64, height: 48, pipelineP95Ms: 80 };

    // First call: nothing has run yet, so the periodic trigger fires.
    controller.maybeRun({ ...base, nowMs: 1000, obstaclePresent: false, directionUncertain: false, sceneChanged: false });
    assert.equal(controller.runs, 1);
    assert.equal(controller.lastTrigger, SegmentationTrigger.PERIODIC);

    // Immediately after, with no trigger, it must skip.
    controller.maybeRun({ ...base, nowMs: 1050, obstaclePresent: false, directionUncertain: false, sceneChanged: false });
    assert.equal(controller.runs, 1);
    assert.equal(controller.skips, 1);

    // An obstacle is a reason to look more carefully.
    controller.maybeRun({ ...base, nowMs: 1100, obstaclePresent: true, directionUncertain: false, sceneChanged: false });
    assert.equal(controller.runs, 2);
    assert.equal(controller.lastTrigger, SegmentationTrigger.OBSTACLE_PRESENT);
});

test("the governor disables the stage when it costs too much", () => {
    // An estimator that pretends to be expensive.
    const slowEstimator = {
        estimate: () => ({ columns: new Float32Array(16).fill(1), blockRow: new Int32Array(16), costMs: 12 })
    };
    const controller = new AdaptiveSegmentationController({
        estimator: slowEstimator,
        budgetMs: 3.5,
        periodicIntervalMs: 0
    });

    for (let i = 0; i < 20 && controller.enabled; i += 1) {
        controller.maybeRun({
            luma: new Uint8Array(64 * 48),
            width: 64,
            height: 48,
            nowMs: 1000 + i * 50,
            obstaclePresent: true,
            directionUncertain: false,
            sceneChanged: false,
            pipelineP95Ms: 80
        });
    }

    assert.equal(controller.enabled, false, "an over-budget stage must switch itself off");
    assert.ok(controller.disabledReason.includes("budget"), controller.disabledReason);
});

test("the governor disables the stage when it dominates a struggling pipeline", () => {
    const estimator = {
        estimate: () => ({ columns: new Float32Array(16).fill(1), blockRow: new Int32Array(16), costMs: 3 })
    };
    const controller = new AdaptiveSegmentationController({
        estimator,
        budgetMs: 5,
        maxCostShare: 0.18,
        periodicIntervalMs: 0
    });

    for (let i = 0; i < 20 && controller.enabled; i += 1) {
        controller.maybeRun({
            luma: new Uint8Array(64 * 48),
            width: 64,
            height: 48,
            nowMs: 1000 + i * 50,
            obstaclePresent: true,
            directionUncertain: false,
            sceneChanged: false,
            // A pipeline already at 12 ms means a 3 ms stage is 25% of it.
            pipelineP95Ms: 12
        });
    }

    assert.equal(controller.enabled, false);
    assert.ok(controller.disabledReason.includes("%"), controller.disabledReason);
});
