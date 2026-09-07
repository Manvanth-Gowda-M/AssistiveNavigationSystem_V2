/**
 * Navigation decisions.
 *
 * The first block is the requirement-71 truth table, driven end to end through
 * the real tracker, corridor, free-space, risk and path-scoring stages via the
 * shared harness. The second block tests hysteresis directly, because "does it
 * refuse to change its mind too easily" needs synthetic path scores to be
 * examined precisely.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    Action,
    DecisionEngine,
    NavigationDecision,
    Urgency
} from "../../src/assistive_navigation/web/static/js/navigation/decisionEngine.js";
import { PipelineHarness } from "../../src/assistive_navigation/web/static/js/testing/pipelineHarness.js";
import { NavigationConfig } from "../../src/assistive_navigation/web/static/js/config/navigationConfig.js";
import { box } from "./helpers.mjs";

const det = (label, b, confidence = 0.88) => ({ label, box: b, confidence });

const CHAIR_CENTRE = box(0.36, 0.48, 0.62, 0.78);
const BLOCK_LEFT = box(0.00, 0.48, 0.30, 0.82);
const BLOCK_RIGHT = box(0.70, 0.48, 1.00, 0.82);

/** Run a static situation until the pipeline settles, then report. */
function settle(detections, frames = 10, context = {}) {
    const harness = new PipelineHarness();
    for (let i = 0; i < frames; i += 1) harness.step(detections, context);
    return {
        harness,
        last: harness.frames[harness.frames.length - 1],
        actions: harness.frames.map((f) => f.decision.action)
    };
}

/* ------------------------------------------- requirement 71 truth table */

test("no obstacle in the path yields CONTINUE and silence", () => {
    const { last, harness } = settle([], 8);
    assert.equal(last.decision.action, Action.CONTINUE);
    assert.equal(harness.utterances.length, 0, "silence is the correct output");
});

test("centre obstacle + left clear + right blocked yields MOVE LEFT", () => {
    const { last } = settle([det("chair", CHAIR_CENTRE), det("bicycle", BLOCK_RIGHT)], 12);
    assert.ok(last.decision.action.includes("LEFT"), `got ${last.decision.action}`);
});

test("centre obstacle + left blocked + right clear yields MOVE RIGHT", () => {
    const { last } = settle([det("chair", CHAIR_CENTRE), det("bicycle", BLOCK_LEFT)], 12);
    assert.ok(last.decision.action.includes("RIGHT"), `got ${last.decision.action}`);
});

test("centre obstacle + both sides blocked yields STOP", () => {
    const { last } = settle(
        [det("chair", CHAIR_CENTRE), det("bicycle", BLOCK_LEFT), det("suitcase", BLOCK_RIGHT)],
        12
    );
    assert.equal(last.decision.action, Action.STOP);
    assert.ok(
        ["blocked_all_directions", "blocked_both_sides", "no_passable_gap", "obstacle_very_near"]
            .includes(last.decision.reason),
        `unexpected reason ${last.decision.reason}`
    );
});

test("the engine never recommends a side that is itself blocked", () => {
    for (const [blocked, blockedBox] of [["LEFT", BLOCK_LEFT], ["RIGHT", BLOCK_RIGHT]]) {
        const { actions } = settle([det("chair", CHAIR_CENTRE), det("bicycle", blockedBox)], 14);
        const offending = actions.filter((a) => a.includes(blocked));
        assert.equal(offending.length, 0, `steered into the blocked ${blocked} side: ${offending.join(",")}`);
    }
});

test("an obstacle outside the corridor does not trigger a manoeuvre", () => {
    const { last } = settle([det("chair", box(0.00, 0.50, 0.18, 0.78))], 10);
    assert.equal(last.decision.action, Action.CONTINUE);
});

/* ------------------------------------------------------ emergency paths */

test("an obstacle at arm's length in the path stops rather than sidesteps", () => {
    // Both flanks are wide open, so a naive engine would happily sidestep.
    const { last } = settle([det("chair", box(0.32, 0.40, 0.68, 0.97), 0.93)], 8);
    assert.equal(last.decision.action, Action.STOP);
    assert.equal(last.decision.urgency, Urgency.CRITICAL);
});

test("a rapidly closing vehicle stops before it fills the frame", () => {
    const harness = new PipelineHarness();
    const frames = [
        [det("car", box(0.44, 0.42, 0.58, 0.56), 0.72)],
        [det("car", box(0.41, 0.40, 0.62, 0.62), 0.80)],
        [det("car", box(0.36, 0.36, 0.68, 0.70), 0.86)],
        [det("car", box(0.28, 0.30, 0.76, 0.80), 0.90)]
    ];
    for (const f of frames) harness.step(f);

    const actions = harness.frames.map((f) => f.decision.action);
    assert.ok(actions.includes(Action.STOP), `never stopped: ${actions.join(",")}`);
    assert.ok(harness.utterances.some((u) => u.text.startsWith("Stop")), "and said so");
});

test("unusable perception asks the user to stop", () => {
    const { last, harness } = settle([], 4, { qualityRequiresStop: true });
    assert.equal(last.decision.action, Action.STOP);
    assert.equal(last.decision.reason, "perception_unreliable");
    assert.ok(harness.utterances.some((u) => u.text.includes("Please stop")));
});

test("poor visibility withholds directions but still warns", () => {
    const { actions, last } = settle(
        [det("chair", CHAIR_CENTRE)],
        10,
        { qualityScore: 0.3, qualityAllowsDirection: false }
    );
    assert.equal(actions.filter((a) => a.includes("LEFT") || a.includes("RIGHT")).length, 0,
        `emitted a direction from untrustworthy frames: ${actions.join(",")}`);
    assert.ok([Action.CAUTION, Action.STOP, Action.CONTINUE].includes(last.decision.action));
});

/* ------------------------------------------------- multi-object reasoning */

test("the engine reasons about the whole scene, not just the worst object", () => {
    // The bicycle on the right is the *lower* risk object, but the right side is
    // the one that is blocked. A single-object engine steers into it.
    const { last } = settle(
        [
            det("chair", CHAIR_CENTRE, 0.92),
            det("bicycle", box(0.68, 0.52, 0.98, 0.80), 0.70)
        ],
        14
    );
    assert.ok(last.decision.action.includes("LEFT"), `got ${last.decision.action}`);
});

test("an unclassified obstacle still produces guidance", () => {
    const { last } = settle([det("obstacle", box(0.34, 0.58, 0.66, 0.90), 0.68)], 12);
    assert.notEqual(last.decision.action, Action.CONTINUE);
});

/* -------------------------------------------------------- decision quality */

test("decision confidence is low when the two sides are equally good", () => {
    const { last } = settle([det("chair", CHAIR_CENTRE)], 12);
    if (last.decision.action.includes("LEFT") || last.decision.action.includes("RIGHT")) {
        assert.ok(last.decision.confidence < 0.85, `unjustified confidence ${last.decision.confidence}`);
    }
});

test("a symmetric situation resolves to one side and stays there", () => {
    const { harness } = settle([det("chair", CHAIR_CENTRE)], 20);
    assert.equal(harness.countDirectionReversals(), 0, "a coin flip must not become oscillation");
});

/* -------------------------------------------------------------- hysteresis */

/** Build a synthetic path-score set. */
function pathScores({ left, center, right, leftPassable = true, rightPassable = true }) {
    const mk = (sector, score, passable) => ({
        sector,
        score,
        adjustedScore: score,
        clearance: score / 100,
        risk: 100 - score,
        passableWidth: passable ? 0.3 : 0.05,
        passable,
        worst: null
    });
    return {
        left: mk("LEFT", left, leftPassable),
        center: mk("CENTER", center, true),
        right: mk("RIGHT", right, rightPassable),
        ordered: [],
        best: "CENTER",
        bounds: { left: 0.3, right: 0.7, centre: 0.5, width: 0.4 }
    };
}

function input({ scores, nowMs, threats = [], emergent = [], freeSpace = { passable: true, bestGapCentre: 0.5 } }) {
    return {
        tracks: threats,
        threats,
        emergent,
        pathScores: scores,
        freeSpace,
        corridor: { bounds: scores.bounds },
        nowMs,
        qualityScore: 1,
        qualityAllowsDirection: true,
        qualityRequiresStop: false
    };
}

test("a held direction is not abandoned for a marginal improvement", () => {
    const engine = new DecisionEngine();
    let t = 0;

    // Establish MOVE RIGHT: centre blocked, right much better than left.
    const establish = pathScores({ left: 40, center: 20, right: 85 });
    for (let i = 0; i < 4; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: establish, nowMs: t }));
    }
    assert.ok(engine.current.action.includes("RIGHT"), `setup failed: ${engine.current.action}`);

    // Now left becomes slightly better than right, well inside the dwell window.
    const marginal = pathScores({ left: 88, center: 20, right: 85 });
    for (let i = 0; i < 6; i += 1) {
        t += 100;
        const decision = engine.evaluate(input({ scores: marginal, nowMs: t }));
        assert.ok(
            decision.action.includes("RIGHT"),
            `flipped on a ${88 - 85}-point advantage at frame ${i}: ${decision.action}`
        );
    }
});

test("a materially better option is adopted after confirmation", () => {
    const engine = new DecisionEngine();
    let t = 0;

    const establish = pathScores({ left: 40, center: 20, right: 85 });
    for (let i = 0; i < 4; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: establish, nowMs: t }));
    }
    assert.ok(engine.current.action.includes("RIGHT"));

    // Left now beats right by far more than the switch margin.
    const decisive = pathScores({ left: 95, center: 20, right: 30 });
    let switched = false;
    for (let i = 0; i < 8; i += 1) {
        t += 100;
        const decision = engine.evaluate(input({ scores: decisive, nowMs: t }));
        if (decision.action.includes("LEFT")) {
            switched = true;
            assert.ok(i >= NavigationConfig.hysteresis.confirmFrames, "switched before confirmation completed");
            break;
        }
    }
    assert.ok(switched, "never adopted the decisively better side");
});

test("an emergency STOP bypasses hysteresis entirely", () => {
    const engine = new DecisionEngine();
    let t = 0;
    const establish = pathScores({ left: 40, center: 20, right: 85 });
    for (let i = 0; i < 4; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: establish, nowMs: t }));
    }
    assert.ok(engine.current.action.includes("RIGHT"));

    // Everything closes in on the very next frame.
    t += 100;
    const blocked = pathScores({ left: 10, center: 5, right: 10, leftPassable: false, rightPassable: false });
    blocked.center.risk = 95;
    const decision = engine.evaluate(input({
        scores: blocked,
        nowMs: t,
        freeSpace: { passable: false, bestGapCentre: 0.5 }
    }));

    assert.equal(decision.action, Action.STOP, "STOP must be immediate");
    assert.equal(decision.held, false);
});

test("opposite reversals need more confirmation than a fresh direction", () => {
    const engine = new DecisionEngine();
    let t = 0;

    // Hold CAUTION first so the dwell timer is not the thing under test.
    const caution = pathScores({ left: 20, center: 25, right: 20, leftPassable: false, rightPassable: false });
    for (let i = 0; i < 3; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: caution, nowMs: t }));
    }

    // A clear right option: adopted after the normal confirmation count.
    const goRight = pathScores({ left: 30, center: 20, right: 90 });
    let framesToAdopt = 0;
    for (let i = 0; i < 6; i += 1) {
        t += 100;
        framesToAdopt += 1;
        if (engine.evaluate(input({ scores: goRight, nowMs: t })).action.includes("RIGHT")) break;
    }
    assert.ok(framesToAdopt <= NavigationConfig.hysteresis.confirmFrames + 1, `took ${framesToAdopt} frames`);

    // Now force a reversal well past the dwell window.
    t += NavigationConfig.hysteresis.holdMs + 200;
    const goLeft = pathScores({ left: 95, center: 20, right: 25 });
    let reversalFrames = 0;
    for (let i = 0; i < 8; i += 1) {
        t += 100;
        reversalFrames += 1;
        if (engine.evaluate(input({ scores: goLeft, nowMs: t })).action.includes("LEFT")) break;
    }
    assert.ok(
        reversalFrames > NavigationConfig.hysteresis.confirmFrames,
        `a left/right reversal took only ${reversalFrames} frames`
    );
});

test("returning to CONTINUE has its own shorter dwell", () => {
    const engine = new DecisionEngine();
    let t = 0;
    const establish = pathScores({ left: 90, center: 20, right: 30 });
    for (let i = 0; i < 4; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: establish, nowMs: t }));
    }
    assert.ok(engine.current.isDirectional);

    const clear = pathScores({ left: 95, center: 95, right: 95 });
    t += 100;
    assert.ok(
        engine.evaluate(input({ scores: clear, nowMs: t })).isDirectional,
        "does not snap back instantly"
    );

    t += NavigationConfig.hysteresis.clearHoldMs + 100;
    assert.equal(engine.evaluate(input({ scores: clear, nowMs: t })).action, Action.CONTINUE);
});

test("path clear is announced only after an obstruction and a stable period", () => {
    const engine = new DecisionEngine();
    let t = 1000;
    const clear = pathScores({ left: 95, center: 95, right: 95 });

    // Never obstructed: nothing to announce.
    for (let i = 0; i < 5; i += 1) {
        t += 200;
        engine.evaluate(input({ scores: clear, nowMs: t }));
    }
    assert.equal(engine.shouldAnnounceClear(t), false);

    // Obstruct, then clear.
    const blocked = pathScores({ left: 90, center: 15, right: 20 });
    for (let i = 0; i < 4; i += 1) {
        t += 200;
        engine.evaluate(input({ scores: blocked, nowMs: t }));
    }
    t += NavigationConfig.hysteresis.clearHoldMs + 100;
    engine.evaluate(input({ scores: clear, nowMs: t }));
    assert.equal(engine.shouldAnnounceClear(t), false, "not immediately");

    t += NavigationConfig.pathClear.stabilityMs + 100;
    engine.evaluate(input({ scores: clear, nowMs: t }));
    assert.equal(engine.shouldAnnounceClear(t), true, "after the stability period");

    engine.markClearAnnounced();
    assert.equal(engine.shouldAnnounceClear(t), false, "and only once");
});

test("suppressed switches are counted for diagnostics", () => {
    const engine = new DecisionEngine();
    let t = 0;
    const establish = pathScores({ left: 30, center: 20, right: 88 });
    for (let i = 0; i < 3; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: establish, nowMs: t }));
    }
    const marginal = pathScores({ left: 90, center: 20, right: 88 });
    for (let i = 0; i < 5; i += 1) {
        t += 100;
        engine.evaluate(input({ scores: marginal, nowMs: t }));
    }
    assert.ok(engine.suppressedSwitches > 0, "hysteresis activity is observable");
});

test("a decision signature is stable under jitter but changes on substance", () => {
    const a = new NavigationDecision({
        action: Action.MOVE_LEFT,
        primaryObstacle: { label: "chair", distanceCategory: "NEAR", risk: 52 }
    });
    const jittered = new NavigationDecision({
        action: Action.MOVE_LEFT,
        primaryObstacle: { label: "chair", distanceCategory: "NEAR", risk: 54 }
    });
    const different = new NavigationDecision({
        action: Action.MOVE_LEFT,
        primaryObstacle: { label: "person", distanceCategory: "NEAR", risk: 52 }
    });

    assert.equal(a.signature(), jittered.signature(), "a 2-point risk wobble is not a new situation");
    assert.notEqual(a.signature(), different.signature(), "a different object is");
});
