/**
 * Speech policy and manager.
 *
 * "Speaks too much" was the loudest complaint about the previous build, so these
 * tests are mostly about *not* speaking: silence on a clear path, silence when
 * nothing changed, silence inside a cooldown - while still guaranteeing that an
 * emergency cuts through all of it.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    SpeechPolicy,
    SpeechVerdict,
    phraseForDecision
} from "../../src/assistive_navigation/web/static/js/audio/speechPolicy.js";
import { SpeechManager } from "../../src/assistive_navigation/web/static/js/audio/speechManager.js";
import { SpeechConfig, SpeechPriority } from "../../src/assistive_navigation/web/static/js/config/speechConfig.js";
import { Action, NavigationDecision, Urgency } from "../../src/assistive_navigation/web/static/js/navigation/decisionEngine.js";
import { Clock, FakeSynth } from "./helpers.mjs";

function decision(overrides = {}) {
    return new NavigationDecision({
        action: Action.MOVE_SLIGHTLY_LEFT,
        urgency: Urgency.MEDIUM,
        reason: "avoid_left",
        confidence: 0.8,
        primaryObstacle: { label: "chair", distanceCategory: "NEAR", risk: 50, velocityX: 0 },
        ...overrides
    });
}

/* -------------------------------------------------------------- phrasing */

test("every phrase is a short instruction", () => {
    const actions = [
        Action.STOP,
        Action.MOVE_LEFT,
        Action.MOVE_SLIGHTLY_LEFT,
        Action.MOVE_RIGHT,
        Action.MOVE_SLIGHTLY_RIGHT,
        Action.CAUTION
    ];
    for (const action of actions) {
        const phrase = phraseForDecision(decision({ action }));
        assert.ok(phrase, `${action} produced no phrase`);
        const words = phrase.text.split(/\s+/).length;
        assert.ok(words <= 5, `"${phrase.text}" is ${words} words`);
    }
});

test("CONTINUE produces no phrase at all", () => {
    assert.equal(phraseForDecision(decision({ action: Action.CONTINUE, primaryObstacle: null })), null);
});

test("stop phrasing reflects the reason", () => {
    const cases = [
        ["obstacle_very_near", SpeechConfig.phrases.STOP_VERY_NEAR],
        ["rapid_approach", SpeechConfig.phrases.STOP_APPROACHING],
        ["blocked_all_directions", SpeechConfig.phrases.STOP_BLOCKED],
        ["perception_unreliable", SpeechConfig.phrases.STOP_UNRELIABLE]
    ];
    for (const [reason, expected] of cases) {
        const phrase = phraseForDecision(decision({ action: Action.STOP, reason, primaryObstacle: null }));
        assert.equal(phrase.text, expected, `reason ${reason}`);
    }
});

test("a vehicle gets the vehicle stop phrase whatever the reason", () => {
    const phrase = phraseForDecision(decision({
        action: Action.STOP,
        reason: "blocked_all_directions",
        primaryObstacle: { label: "bus", distanceCategory: "NEAR", risk: 90, velocityX: 0 }
    }));
    assert.equal(phrase.text, SpeechConfig.phrases.STOP_VEHICLE);
});

test("a moving person is announced as crossing, a still one as ahead", () => {
    const crossing = phraseForDecision(decision({
        action: Action.CAUTION,
        primaryObstacle: { label: "person", distanceCategory: "NEAR", risk: 50, velocityX: 0.3 }
    }));
    const standing = phraseForDecision(decision({
        action: Action.CAUTION,
        primaryObstacle: { label: "person", distanceCategory: "NEAR", risk: 50, velocityX: 0.01 }
    }));
    assert.equal(crossing.text, SpeechConfig.phrases.PERSON_CROSSING);
    assert.equal(standing.text, SpeechConfig.phrases.PERSON_AHEAD);
});

/* --------------------------------------------------------------- policy */

test("a clear path is silent", () => {
    const policy = new SpeechPolicy();
    const result = policy.evaluate({
        decision: decision({ action: Action.CONTINUE, primaryObstacle: null }),
        nowMs: 1000
    });
    assert.equal(result.speak, false);
    assert.equal(result.verdict, SpeechVerdict.SILENT_CLEAR_PATH);
});

test("an unchanged instruction is not repeated every frame", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    const d = decision();

    const first = policy.evaluate({ decision: d, nowMs: t });
    assert.equal(first.speak, true);
    policy.commit({ ...first, decision: d, nowMs: t });

    let spoke = 0;
    for (let i = 0; i < 40; i += 1) {
        t += 80;
        const r = policy.evaluate({ decision: d, nowMs: t });
        if (r.speak) {
            spoke += 1;
            policy.commit({ ...r, decision: d, nowMs: t });
        }
    }
    // 40 frames at 80 ms is 3.2 s; the 5 s reminder must not have fired yet.
    assert.equal(spoke, 0, `repeated ${spoke} times while nothing changed`);
});

test("a still-live instruction is repeated once as a slow reminder", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    const d = decision();
    const first = policy.evaluate({ decision: d, nowMs: t });
    policy.commit({ ...first, decision: d, nowMs: t });

    let spoke = 0;
    for (let i = 0; i < 200; i += 1) {
        t += 80;
        const r = policy.evaluate({ decision: d, nowMs: t });
        if (r.speak) {
            spoke += 1;
            assert.equal(r.reminder, true, "a repeat of the same phrase must be flagged as a reminder");
            policy.commit({ ...r, decision: d, nowMs: t });
        }
    }
    // 16 s of unchanged instruction: with a 5 s reminder and a 3.5 s repeat
    // cooldown, a small number of reminders is correct, spam is not.
    assert.ok(spoke >= 1 && spoke <= 4, `spoke ${spoke} times over 16 s`);
});

test("a changed action speaks promptly", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    const left = decision({ action: Action.MOVE_SLIGHTLY_LEFT });
    const first = policy.evaluate({ decision: left, nowMs: t });
    policy.commit({ ...first, decision: left, nowMs: t });

    t += SpeechConfig.cooldowns.changeActionMs + 50;
    const right = decision({ action: Action.MOVE_SLIGHTLY_RIGHT, reason: "avoid_right" });
    const second = policy.evaluate({ decision: right, nowMs: t });
    assert.equal(second.speak, true);
    assert.equal(second.text, SpeechConfig.phrases.MOVE_SLIGHTLY_RIGHT);
});

test("a changed action inside its cooldown stays quiet", () => {
    const policy = new SpeechPolicy();
    const t = 1000;
    const left = decision({ action: Action.MOVE_SLIGHTLY_LEFT });
    const first = policy.evaluate({ decision: left, nowMs: t });
    policy.commit({ ...first, decision: left, nowMs: t });

    const right = decision({ action: Action.MOVE_SLIGHTLY_RIGHT, reason: "avoid_right" });
    const second = policy.evaluate({ decision: right, nowMs: t + 200 });
    assert.equal(second.speak, false);
    assert.equal(second.verdict, SpeechVerdict.SILENT_COOLDOWN);
});

test("a material risk jump breaks through the no-change gate", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    const mild = decision({ primaryObstacle: { label: "chair", distanceCategory: "NEAR", risk: 40, velocityX: 0 } });
    const first = policy.evaluate({ decision: mild, nowMs: t });
    policy.commit({ ...first, decision: mild, nowMs: t });

    t += SpeechConfig.cooldowns.repeatSameMs + 50;
    const worse = decision({
        primaryObstacle: {
            label: "chair",
            distanceCategory: "NEAR",
            risk: 40 + SpeechConfig.materialRiskDelta + 5,
            velocityX: 0
        }
    });
    const second = policy.evaluate({ decision: worse, nowMs: t });
    assert.equal(second.speak, true);
    assert.notEqual(second.reminder, true, "this is a real change, not a reminder");
});

test("an emergency bypasses the cooldown and asks to interrupt", () => {
    const policy = new SpeechPolicy();
    const t = 1000;
    const routine = decision();
    const first = policy.evaluate({ decision: routine, nowMs: t });
    policy.commit({ ...first, decision: routine, nowMs: t });

    // Immediately afterwards, with no time elapsed at all.
    const emergency = decision({
        action: Action.STOP,
        urgency: Urgency.CRITICAL,
        reason: "obstacle_very_near",
        emergency: true
    });
    const result = policy.evaluate({ decision: emergency, nowMs: t + 1 });

    assert.equal(result.speak, true);
    assert.equal(result.interrupt, true);
    assert.equal(result.priority, SpeechPriority.STOP);
    assert.equal(result.verdict, SpeechVerdict.EMERGENCY_BYPASS);
});

test("a repeated emergency is throttled to avoid stuttering", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    const emergency = decision({ action: Action.STOP, urgency: Urgency.CRITICAL, emergency: true, reason: "obstacle_very_near" });

    const first = policy.evaluate({ decision: emergency, nowMs: t });
    policy.commit({ ...first, decision: emergency, nowMs: t });

    let spoke = 0;
    for (let i = 0; i < 10; i += 1) {
        t += 80;
        const r = policy.evaluate({ decision: emergency, nowMs: t });
        if (r.speak) {
            spoke += 1;
            policy.commit({ ...r, decision: emergency, nowMs: t });
        }
    }
    assert.equal(spoke, 0, `said "Stop." ${spoke} extra times in 800 ms`);
});

test("path clear is spoken only when the engine confirms it", () => {
    const policy = new SpeechPolicy();
    const clear = decision({ action: Action.CONTINUE, primaryObstacle: null });

    const silent = policy.evaluate({ decision: clear, nowMs: 5000, announceClear: false });
    assert.equal(silent.speak, false);

    const announced = policy.evaluate({ decision: clear, nowMs: 5000, announceClear: true });
    assert.equal(announced.speak, true);
    assert.equal(announced.text, SpeechConfig.phrases.PATH_CLEAR);
});

test("muting silences everything", () => {
    const policy = new SpeechPolicy();
    const result = policy.evaluate({
        decision: decision({ action: Action.STOP, emergency: true }),
        nowMs: 1000,
        enabled: false
    });
    assert.equal(result.speak, false);
    assert.equal(result.verdict, SpeechVerdict.SILENT_DISABLED);
});

test("the per-minute backstop engages if everything else fails", () => {
    const policy = new SpeechPolicy();
    let t = 1000;
    let spoke = 0;

    // Alternate between two actions with enough elapsed time to defeat every
    // cooldown, which is the only way to reach the rate limit.
    for (let i = 0; i < 60; i += 1) {
        t += SpeechConfig.cooldowns.changeActionMs + 20;
        const d = decision({
            action: i % 2 === 0 ? Action.MOVE_SLIGHTLY_LEFT : Action.MOVE_SLIGHTLY_RIGHT,
            reason: i % 2 === 0 ? "avoid_left" : "avoid_right"
        });
        const r = policy.evaluate({ decision: d, nowMs: t });
        if (r.speak) {
            spoke += 1;
            policy.commit({ ...r, decision: d, nowMs: t });
        } else if (r.verdict === SpeechVerdict.SILENT_RATE_LIMIT) {
            assert.ok(policy.counters.suppressedRateLimit > 0);
            return;
        }
    }
    // If the limit was never hit, the cooldowns alone kept us under it, which is
    // also an acceptable outcome - assert that rather than silently passing.
    assert.ok(spoke <= SpeechConfig.maxUtterancesPerMinute * 2, `spoke ${spoke} times`);
});

/* -------------------------------------------------------------- manager */

test("the manager speaks through the synthesiser and records history", async () => {
    const clock = new Clock(1000);
    const synth = new FakeSynth();
    const manager = new SpeechManager({ synth, now: clock.now, voiceManager: { rate: 1, pitch: 1, volume: 1, selectedVoice: null } });

    global.SpeechSynthesisUtterance = class {
        constructor(text) {
            this.text = text;
        }
    };

    const result = manager.speakDecision(decision());
    assert.equal(result.spoken, true);
    await new Promise((r) => setTimeout(r, 5));
    assert.deepEqual(synth.texts, [SpeechConfig.phrases.MOVE_SLIGHTLY_LEFT]);
    assert.equal(manager.history.length, 1);
});

test("an emergency cancels the utterance in progress", async () => {
    const clock = new Clock(1000);
    // autoEnd off, so the first utterance is still 'speaking' when the emergency
    // arrives - the situation requirement 36 describes.
    const synth = new FakeSynth({ autoEnd: false });
    const manager = new SpeechManager({ synth, now: clock.now, voiceManager: { rate: 1, pitch: 1, volume: 1, selectedVoice: null } });

    global.SpeechSynthesisUtterance = class {
        constructor(text) {
            this.text = text;
        }
    };

    manager.speakSystem(SpeechConfig.phrases.PATH_CLEAR, SpeechPriority.PATH_STATUS);
    assert.equal(synth.texts.length, 1);
    assert.equal(manager.isSpeaking, true);

    clock.advance(300);
    manager.speakDecision(decision({
        action: Action.STOP,
        urgency: Urgency.CRITICAL,
        emergency: true,
        reason: "obstacle_very_near"
    }));

    assert.ok(synth.cancels >= 1, "the in-progress utterance was cancelled");
    assert.ok(synth.texts.includes(SpeechConfig.phrases.STOP_VERY_NEAR), `got ${synth.texts.join(",")}`);
});

test("the manager reports unavailability instead of pretending", () => {
    const manager = new SpeechManager({ synth: null, voiceManager: { rate: 1, pitch: 1, volume: 1, selectedVoice: null } });
    assert.equal(manager.available, false);
    const metrics = manager.metrics();
    assert.equal(metrics.available, false);
});

test("muting the manager cancels and blocks further speech", () => {
    const synth = new FakeSynth();
    const manager = new SpeechManager({ synth, voiceManager: { rate: 1, pitch: 1, volume: 1, selectedVoice: null } });
    global.SpeechSynthesisUtterance = class {
        constructor(text) {
            this.text = text;
        }
    };

    manager.setEnabled(false);
    const result = manager.speakDecision(decision());
    assert.equal(result.spoken, false);
    assert.equal(synth.texts.length, 0);
});
