/**
 * Simulation and stability tests.
 *
 * Requirement 72: manual testing alone misses intermittent failures. These tests
 * replay long scripted sequences through the full reasoning stack and assert on
 * aggregate behaviour - continuity, stability, speech volume and bounded state -
 * which is where the failures that ruin a walking demo actually live.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { SCENARIOS, generateWalkSequence } from "../../src/assistive_navigation/web/static/js/testing/scenarios.js";
import { PipelineHarness } from "../../src/assistive_navigation/web/static/js/testing/pipelineHarness.js";
import { runScenario } from "../../src/assistive_navigation/web/static/js/ui/scenarioRunner.js";
import { Action } from "../../src/assistive_navigation/web/static/js/navigation/decisionEngine.js";
import { SpeechConfig } from "../../src/assistive_navigation/web/static/js/config/speechConfig.js";

/* ------------------------------------------------------ scenario suite */

test("every scripted scenario meets its expectations", () => {
    const failures = [];

    for (const scenario of SCENARIOS) {
        const { verdict } = runScenario(scenario);
        if (!verdict.passed) {
            failures.push(
                `${scenario.id}: ` + verdict.checks
                    .filter((c) => !c.ok)
                    .map((c) => `${c.label} [${c.detail}]`)
                    .join("; ")
            );
        }
    }

    assert.deepEqual(failures, [], `scenario failures:\n${failures.join("\n")}`);
});

test("the scenario suite covers the required decision matrix", () => {
    const ids = new Set(SCENARIOS.map((s) => s.id));
    for (const required of [
        "clear-path",
        "centre-obstacle-left-clear",
        "centre-obstacle-right-clear",
        "centre-obstacle-both-blocked",
        "person-crossing",
        "rapid-approach",
        "very-near-obstacle",
        "unknown-obstacle",
        "obstacle-then-clear",
        "low-visibility"
    ]) {
        assert.ok(ids.has(required), `missing scenario: ${required}`);
    }
});

/* --------------------------------------------------- detection continuity */

test("a persistent obstacle produces one continuous track, not a stream of new ones", () => {
    const harness = new PipelineHarness();
    const det = [{ label: "chair", box: [0.38, 0.50, 0.60, 0.80], confidence: 0.88 }];
    for (let i = 0; i < 60; i += 1) harness.step(det);

    const ids = new Set();
    for (const frame of harness.frames) {
        for (const track of frame.tracks) ids.add(track.id);
    }
    assert.equal(ids.size, 1, `identity churned across ${ids.size} track ids`);
    assert.equal(harness.tracker.totalCreated, 1);
});

test("a person walking across the scene is tracked continuously then released", () => {
    const harness = new PipelineHarness();

    for (let i = 0; i < 12; i += 1) {
        const x = 0.02 + i * 0.075;
        harness.step([{ label: "person", box: [x, 0.30, x + 0.13, 0.80], confidence: 0.90 }]);
    }
    const midRunIds = new Set(harness.frames.slice(4).flatMap((f) => f.tracks.map((t) => t.id)));
    assert.equal(midRunIds.size, 1, "one identity throughout the crossing");

    for (let i = 0; i < 14; i += 1) harness.step([]);
    const finalFrame = harness.frames[harness.frames.length - 1];
    assert.equal(finalFrame.tracks.length, 0, "the track is released once they are gone");
    assert.equal(finalFrame.decision.action, Action.CONTINUE);
});

test("detector flicker does not break track continuity", () => {
    const harness = new PipelineHarness();
    const det = [{ label: "chair", box: [0.38, 0.50, 0.60, 0.80], confidence: 0.88 }];

    // Drop every fourth frame, which is a realistic detector miss rate while
    // walking.
    for (let i = 0; i < 60; i += 1) harness.step(i % 4 === 3 ? [] : det);

    const ids = new Set(harness.frames.flatMap((f) => f.tracks.map((t) => t.id)));
    assert.ok(ids.size <= 2, `flicker created ${ids.size} identities`);
    assert.ok(harness.frames[harness.frames.length - 1].tracks.length >= 1, "the obstacle is still known");
});

/* ---------------------------------------------------- decision stability */

test("a long clear walk is completely silent", () => {
    const harness = new PipelineHarness();
    for (let i = 0; i < 300; i += 1) harness.step([]);
    assert.equal(harness.utterances.length, 0);
    assert.equal(harness.countActionSwitches(), 0);
});

test("a long synthetic walk never oscillates left and right", () => {
    const harness = new PipelineHarness();
    for (const frame of generateWalkSequence(900)) harness.step(frame);

    const reversals = harness.countDirectionReversals();
    // Obstacles genuinely move from one side to the other over 15 cycles, so some
    // reversals are correct. What must not happen is a reversal every few frames.
    assert.ok(reversals < 40, `${reversals} left/right reversals over 900 frames`);
    assert.ok(
        harness.countActionSwitches() < harness.frames.length * 0.2,
        `${harness.countActionSwitches()} action switches in ${harness.frames.length} frames`
    );
});

test("speech volume over a long walk stays within a usable budget", () => {
    const harness = new PipelineHarness({ frameIntervalMs: 80 });
    const frames = generateWalkSequence(900);
    for (const frame of frames) harness.step(frame);

    const walkSeconds = (frames.length * 80) / 1000;
    const perMinute = harness.utterances.length / (walkSeconds / 60);
    const emergencies = harness.utterances.filter((u) => u.emergency);
    const routine = harness.utterances.filter((u) => !u.emergency);
    const routinePerMinute = routine.length / (walkSeconds / 60);

    assert.ok(harness.utterances.length > 0, "a walk with obstacles must produce some guidance");

    // Routine speech is held to the configured chatter budget.
    assert.ok(
        routinePerMinute <= SpeechConfig.maxUtterancesPerMinute,
        `${routinePerMinute.toFixed(1)} routine utterances per minute exceeds the ${SpeechConfig.maxUtterancesPerMinute} budget`
    );

    // Emergencies are exempt from that budget by design, but must still be
    // spaced by the anti-stutter floor. This sequence walks the user into an
    // obstacle every ten seconds, which is far denser than a real walk, so the
    // total is expected to sit above the routine budget.
    assert.ok(perMinute <= 25, `${perMinute.toFixed(1)} total utterances per minute is too chatty`);

    for (let i = 1; i < emergencies.length; i += 1) {
        const gap = emergencies[i].atMs - emergencies[i - 1].atMs;
        assert.ok(
            gap >= SpeechConfig.cooldowns.emergencyMs,
            `two emergency utterances only ${gap} ms apart`
        );
    }
});

test("the rate backstop never starves directional guidance after a hazard", () => {
    const harness = new PipelineHarness({ frameIntervalMs: 80 });
    for (const frame of generateWalkSequence(900)) harness.step(frame);

    // If emergencies counted against the chatter budget they would fill the
    // window and suppress the instruction the user needs next.
    assert.equal(
        harness.speechPolicy.counters.suppressedRateLimit,
        0,
        "the backstop engaged, which means routine speech was being starved"
    );
    assert.ok(
        harness.utterances.some((u) => u.text.startsWith("Move")),
        "directional guidance was still delivered"
    );
});

test("identical inputs produce identical outputs", () => {
    const frames = generateWalkSequence(300);

    const runOnce = () => {
        const harness = new PipelineHarness();
        for (const frame of frames) harness.step(frame);
        return {
            actions: harness.frames.map((f) => f.decision.action).join(","),
            speech: harness.utterances.map((u) => `${u.atMs}:${u.text}`).join("|")
        };
    };

    const a = runOnce();
    const b = runOnce();
    assert.equal(a.actions, b.actions, "the decision sequence is deterministic");
    assert.equal(a.speech, b.speech, "the speech sequence is deterministic");
});

/* ------------------------------------------------------------ stability */

test("a five-minute equivalent walk keeps its state bounded", () => {
    // 12 inference FPS for 5 minutes is 3,600 frames.
    const harness = new PipelineHarness({ frameIntervalMs: 83 });
    const frames = generateWalkSequence(3600);

    let peakTracks = 0;
    for (let i = 0; i < frames.length; i += 1) {
        const record = harness.step(frames[i]);
        peakTracks = Math.max(peakTracks, record.tracks.length);

        // Per-track history is capped, so no single track can grow without bound.
        for (const track of harness.tracker.tracks.values()) {
            assert.ok(track.history.length <= harness.tracker.cfg.historyLength,
                `track ${track.id} history grew to ${track.history.length}`);
        }
    }

    assert.ok(peakTracks <= 8, `peak concurrent tracks ${peakTracks}`);
    assert.ok(harness.tracker.count <= 8, `${harness.tracker.count} tracks left at the end`);

    // Turnover: almost everything created was also destroyed. This is the
    // property that distinguishes a bounded working set from a slow leak.
    const created = harness.tracker.totalCreated;
    const expired = harness.tracker.totalExpired;
    assert.ok(created > 30, `only ${created} tracks were created; the sequence is not exercising turnover`);
    assert.ok(
        expired >= created - 8,
        `${created} tracks created but only ${expired} expired`
    );
});

test("a ten-minute equivalent walk with turns and lighting changes stays healthy", () => {
    const harness = new PipelineHarness({ frameIntervalMs: 83 });
    const frames = generateWalkSequence(7200);

    let turns = 0;
    let lowVisibilityFrames = 0;

    for (let i = 0; i < frames.length; i += 1) {
        // A turn every ~25 s, and a stretch of poor visibility every ~40 s.
        const isTurn = i % 300 === 0 && i > 0;
        const poorVisibility = (i % 480) > 455;

        if (isTurn) turns += 1;
        if (poorVisibility) lowVisibilityFrames += 1;

        harness.step(frames[i], {
            trustTracks: !isTurn,
            qualityScore: poorVisibility ? 0.3 : 1,
            qualityAllowsDirection: !poorVisibility
        });
    }

    assert.ok(turns > 20, `only simulated ${turns} turns`);
    assert.ok(lowVisibilityFrames > 300, `only simulated ${lowVisibilityFrames} poor frames`);

    // Bounded state.
    assert.ok(harness.tracker.count <= 8, `${harness.tracker.count} tracks remain`);
    assert.ok(harness.speechPolicy._recent.length <= 60, "the speech rate window is bounded");

    // No permanent freeze: the system must still be producing decisions at the
    // end, and must have produced a variety of them.
    const tail = harness.frames.slice(-200).map((f) => f.decision.action);
    assert.ok(tail.length === 200);
    assert.ok(new Set(harness.frames.map((f) => f.decision.action)).size >= 2,
        "the system was not stuck in a single state for ten minutes");

    // No confident directions from untrustworthy frames.
    for (const frame of harness.frames) {
        if (frame.decision.reason === "low_visibility_direction_withheld") {
            assert.ok(!frame.decision.action.includes("LEFT") && !frame.decision.action.includes("RIGHT"));
        }
    }
});

test("recovery from a total detection blackout is automatic", () => {
    const harness = new PipelineHarness();
    const det = [{ label: "chair", box: [0.38, 0.50, 0.60, 0.80], confidence: 0.88 }];

    for (let i = 0; i < 20; i += 1) harness.step(det);
    const beforeBlackout = harness.frames[harness.frames.length - 1].decision.action;
    assert.notEqual(beforeBlackout, Action.CONTINUE);

    // The detector goes blind for two seconds.
    for (let i = 0; i < 25; i += 1) harness.step([]);
    assert.equal(harness.frames[harness.frames.length - 1].decision.action, Action.CONTINUE);
    assert.equal(harness.tracker.count, 0, "stale obstacles were not carried forward");

    // Detection returns; guidance must come back without intervention.
    for (let i = 0; i < 20; i += 1) harness.step(det);
    assert.notEqual(harness.frames[harness.frames.length - 1].decision.action, Action.CONTINUE);
});

test("obstacles appearing and clearing repeatedly do not leak speech state", () => {
    const harness = new PipelineHarness();
    const det = [
        { label: "chair", box: [0.38, 0.50, 0.60, 0.80], confidence: 0.9 },
        { label: "bicycle", box: [0.72, 0.50, 1.00, 0.82], confidence: 0.85 }
    ];

    let pathClearCount = 0;
    for (let cycle = 0; cycle < 12; cycle += 1) {
        for (let i = 0; i < 12; i += 1) harness.step(det);
        for (let i = 0; i < 40; i += 1) harness.step([]);
        pathClearCount = harness.utterances.filter((u) => u.text === "Path clear.").length;
    }

    // One "Path clear." per obstruction episode, not one per clear frame.
    assert.ok(pathClearCount >= 1, "clearance is announced");
    assert.ok(pathClearCount <= 12, `announced clearance ${pathClearCount} times in 12 episodes`);
    assert.ok(harness.speechPolicy.counters.suppressedNoChange > 0, "suppression is doing work");
});
