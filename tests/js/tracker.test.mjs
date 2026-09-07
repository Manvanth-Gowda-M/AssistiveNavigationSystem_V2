/**
 * Temporal tracker.
 *
 * The behaviours under test are the ones that decide whether guidance is stable:
 * confirmation gating, identity preservation, coasting with a motion model,
 * expiry, and the emergency channel that must bypass smoothing.
 */

import test from "node:test";
import assert from "node:assert/strict";

import { ObjectTracker } from "../../src/assistive_navigation/web/static/js/vision/tracker.js";
import { VisionConfig } from "../../src/assistive_navigation/web/static/js/config/visionConfig.js";
import { box, detection } from "./helpers.mjs";

const STEP = 80;

function feed(tracker, framesOfDetections, startMs = 1000, context = {}) {
    let t = startMs;
    let last = null;
    for (const frame of framesOfDetections) {
        last = tracker.update(frame.map((d) => ({ ...d, box: [...d.box] })), t, context);
        t += STEP;
    }
    return { last, endMs: t };
}

test("a track needs the configured number of frames before it is confirmed", () => {
    const tracker = new ObjectTracker();
    const det = [detection("chair", box(0.4, 0.5, 0.6, 0.8))];

    const first = tracker.update(det, 1000);
    assert.equal(first.tracks.length, 1);
    assert.equal(first.confirmed.length, 0, "one frame is not evidence");

    tracker.update(det, 1080);
    const third = tracker.update(det, 1160);
    assert.equal(VisionConfig.tracking.confirmFrames, 3);
    assert.equal(third.confirmed.length, 1, "confirmed on the third consistent frame");
});

test("detections below the acceptance floor never open a track", () => {
    const tracker = new ObjectTracker();
    const weak = [detection("chair", box(0.4, 0.5, 0.6, 0.8), VisionConfig.detection.acceptThreshold - 0.05)];
    const { last } = feed(tracker, [weak, weak, weak, weak]);
    assert.equal(last.tracks.length, 0);
});

test("a moving object keeps one identity and gains a velocity estimate", () => {
    const tracker = new ObjectTracker();
    const frames = [];
    for (let i = 0; i < 6; i += 1) {
        const x = 0.10 + i * 0.06;
        frames.push([detection("person", box(x, 0.3, x + 0.14, 0.8), 0.9)]);
    }
    const { last } = feed(tracker, frames);

    assert.equal(last.tracks.length, 1, "one object, one track");
    const track = last.tracks[0];
    assert.equal(track.id, 1);
    assert.ok(track.velocityX > 0.2, `expected rightward motion, got ${track.velocityX}`);
    assert.ok(Math.abs(track.velocityY) < 0.1, "no vertical drift");
});

test("two objects crossing do not swap identities", () => {
    const tracker = new ObjectTracker();
    const frames = [];
    // A person moving right and a chair sitting still, passing close by.
    for (let i = 0; i < 8; i += 1) {
        const px = 0.06 + i * 0.08;
        frames.push([
            detection("person", box(px, 0.30, px + 0.12, 0.80), 0.9),
            detection("chair", box(0.44, 0.52, 0.62, 0.80), 0.88)
        ]);
    }
    const { last } = feed(tracker, frames);

    const person = last.tracks.find((t) => t.label === "person");
    const chair = last.tracks.find((t) => t.label === "chair");
    assert.ok(person && chair, "both tracks survive");
    assert.equal(person.id, 1);
    assert.equal(chair.id, 2);
    assert.ok(Math.abs(chair.velocityX) < 0.06, "the stationary object stays stationary");
});

test("different known labels are never associated with each other", () => {
    const tracker = new ObjectTracker();
    const chair = [detection("chair", box(0.4, 0.5, 0.6, 0.8), 0.9)];
    feed(tracker, [chair, chair, chair]);

    // Same position, different class: must open a new track, not hijack the old.
    const car = [detection("car", box(0.4, 0.5, 0.6, 0.8), 0.9)];
    const result = tracker.update(car, 1400);
    assert.equal(result.tracks.length, 2);
    assert.deepEqual(result.tracks.map((t) => t.label).sort(), ["car", "chair"]);
});

test("an unknown obstacle is upgraded when a class arrives", () => {
    const tracker = new ObjectTracker();
    const unknown = [detection("obstacle", box(0.4, 0.5, 0.6, 0.8), 0.7)];
    feed(tracker, [unknown, unknown]);

    const named = [detection("chair", box(0.4, 0.5, 0.6, 0.8), 0.8)];
    const result = tracker.update(named, 1200);
    assert.equal(result.tracks.length, 1, "the track is reused, not duplicated");
    assert.equal(result.tracks[0].label, "chair");
});

test("a lost track coasts along its velocity, then expires", () => {
    const tracker = new ObjectTracker();
    const frames = [];
    for (let i = 0; i < 5; i += 1) {
        const x = 0.20 + i * 0.05;
        frames.push([detection("person", box(x, 0.3, x + 0.14, 0.8), 0.9)]);
    }
    const { last, endMs } = feed(tracker, frames);
    // Snapshot primitives: tracks are mutated in place, so holding the object
    // reference would compare a value against itself.
    const centreBefore = last.tracks[0].centerX;
    const confidenceBefore = last.tracks[0].confidence;

    const coasted = tracker.update([], endMs);
    assert.equal(coasted.tracks.length, 1, "not deleted immediately");
    assert.ok(coasted.tracks[0].centerX > centreBefore, "the box keeps moving while coasting");
    assert.ok(coasted.tracks[0].confidence < confidenceBefore, "confidence decays");

    // Coast until it is gone.
    let t = endMs;
    for (let i = 0; i < VisionConfig.tracking.maxCoastFrames + 3; i += 1) {
        t += STEP;
        tracker.update([], t);
    }
    assert.equal(tracker.count, 0, "the track is eventually removed");
});

test("a track expires on wall-clock timeout even without further updates", () => {
    const tracker = new ObjectTracker();
    const det = [detection("chair", box(0.4, 0.5, 0.6, 0.8), 0.9)];
    feed(tracker, [det, det, det]);

    // One update far in the future: the coast-frame count is 1 but the wall
    // clock has passed the timeout, which is the guard against a stalled
    // pipeline holding ghost obstacles.
    tracker.update([], 1000 + VisionConfig.tracking.maxCoastMs + 500);
    assert.equal(tracker.count, 0);
});

test("a growing centred object reports approach and rapid approach", () => {
    const tracker = new ObjectTracker();
    const frames = [];
    for (let i = 0; i < 5; i += 1) {
        const half = 0.05 + i * 0.05;
        const y2 = 0.55 + i * 0.07;
        frames.push([detection("car", box(0.5 - half, y2 - 2 * half, 0.5 + half, y2), 0.9)]);
    }
    const { last } = feed(tracker, frames);
    const track = last.tracks[0];
    assert.ok(track.approaching, `expected approaching, growth=${track.areaGrowthRate}`);
    assert.ok(track.rapidlyApproaching, `expected rapid approach, growth=${track.areaGrowthRate}`);
});

test("a shrinking object is not treated as approaching", () => {
    const tracker = new ObjectTracker();
    const frames = [];
    for (let i = 0; i < 5; i += 1) {
        const half = 0.25 - i * 0.04;
        frames.push([detection("car", box(0.5 - half, 0.4, 0.5 + half, 0.85), 0.9)]);
    }
    const { last } = feed(tracker, frames);
    assert.equal(last.tracks[0].approaching, false);
});

test("the emergency channel fires without waiting for confirmation", () => {
    const tracker = new ObjectTracker();
    // Large, centred, low in frame, high confidence: one frame is enough.
    const huge = [detection("person", box(0.25, 0.20, 0.75, 0.95), 0.93)];
    const result = tracker.update(huge, 1000);

    assert.equal(result.confirmed.length, 0, "still unconfirmed");
    assert.equal(result.emergent.length, 1, "but flagged as an emergency");
});

test("the emergency channel ignores distant or low-confidence objects", () => {
    const tracker = new ObjectTracker();

    const distant = tracker.update([detection("person", box(0.46, 0.40, 0.54, 0.50), 0.95)], 1000);
    assert.equal(distant.emergent.length, 0, "small and high in frame is not an emergency");

    const tracker2 = new ObjectTracker();
    const unsure = tracker2.update(
        [detection("person", box(0.25, 0.20, 0.75, 0.95), VisionConfig.detection.emergencyThreshold - 0.05)],
        1000
    );
    assert.equal(unsure.emergent.length, 0, "below the emergency confidence floor");
});

test("an off-centre large object is not an emergency", () => {
    const tracker = new ObjectTracker();
    const offCentre = tracker.update([detection("car", box(0.00, 0.25, 0.26, 0.95), 0.95)], 1000);
    assert.equal(offCentre.emergent.length, 0);
});

test("a camera turn decays confidence and revokes confirmation", () => {
    const tracker = new ObjectTracker();
    const det = [detection("chair", box(0.4, 0.5, 0.6, 0.8), 0.9)];
    const { last, endMs } = feed(tracker, [det, det, det, det]);
    assert.equal(last.confirmed.length, 1);
    const confidenceBefore = last.tracks[0].confidence;
    const stabilityBefore = last.tracks[0].stability;

    const afterTurn = tracker.update(det, endMs, { trustTracks: false });
    assert.equal(afterTurn.confirmed.length, 0, "confirmation is revoked");
    assert.ok(
        afterTurn.tracks[0].confidence < confidenceBefore,
        "confidence is decayed so stale geometry carries less weight"
    );
    assert.ok(afterTurn.tracks[0].stability < stabilityBefore, "stability drops too");

    // It must be able to earn confirmation back once the new view is consistent.
    let t = endMs;
    for (let i = 0; i < 3; i += 1) {
        t += STEP;
        tracker.update(det, t);
    }
    assert.equal(tracker.tracks.get(1).confirmed, true, "re-confirms after re-establishing");
});

test("stability rises with consistent association", () => {
    const tracker = new ObjectTracker();
    const det = [detection("chair", box(0.4, 0.5, 0.6, 0.8), 0.9)];
    const early = tracker.update(det, 1000).tracks[0].stability;
    for (let i = 1; i < 10; i += 1) tracker.update(det, 1000 + i * STEP);
    const later = tracker.tracks.get(1).stability;
    assert.ok(later > early, `${later} should exceed ${early}`);
    assert.ok(later <= 1);
});

test("track count stays bounded over a long noisy run", () => {
    const tracker = new ObjectTracker();
    let t = 1000;
    for (let i = 0; i < 1200; i += 1) {
        t += STEP;
        // A persistent object plus a randomly placed transient, deterministically
        // generated so the test is reproducible.
        const jitter = ((i * 37) % 100) / 100;
        tracker.update(
            [
                detection("chair", box(0.45, 0.55, 0.60, 0.80), 0.9),
                detection("person", box(jitter * 0.7, 0.35, jitter * 0.7 + 0.12, 0.75), 0.85)
            ],
            t
        );
    }
    assert.ok(tracker.count <= 8, `track table grew to ${tracker.count}`);
    assert.ok(tracker.totalExpired > 100, "transients were expired, not accumulated");
});
