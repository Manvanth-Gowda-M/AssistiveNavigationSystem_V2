/**
 * Free-space estimation and the walking corridor.
 *
 * This is the layer that turns "objects exist" into "where can I walk", so the
 * tests here focus on the distinction the old system could not make: an object
 * beside you versus an object in front of you.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    FreeSpaceEstimator,
    aggregateClearance,
    bottomYToFreeDepth,
    widestPassableRun
} from "../../src/assistive_navigation/web/static/js/perception/freeSpace.js";
import { CorridorModel } from "../../src/assistive_navigation/web/static/js/navigation/corridorModel.js";
import { NavigationConfig } from "../../src/assistive_navigation/web/static/js/config/navigationConfig.js";
import { box } from "./helpers.mjs";

const GROUND_TOP = NavigationConfig.freeSpace.groundBandTop;

/** Minimal track-shaped object; the estimator only reads geometry and evidence. */
function fakeTrack({ id = 1, label = "chair", b, confidence = 0.9, visibility = 1 }) {
    return {
        id,
        label,
        confidence,
        visibility,
        box: [...b],
        bottomY: b[3],
        area: (b[2] - b[0]) * (b[3] - b[1])
    };
}

test("bottomYToFreeDepth maps ground contact to remaining depth", () => {
    assert.equal(bottomYToFreeDepth(1.0, GROUND_TOP), 0, "at your feet: no depth left");
    assert.equal(bottomYToFreeDepth(GROUND_TOP, GROUND_TOP), 1, "at the far boundary: fully open");
    const mid = bottomYToFreeDepth((1 + GROUND_TOP) / 2, GROUND_TOP);
    assert.ok(mid > 0.45 && mid < 0.55, `expected mid-range, got ${mid}`);
});

test("widestPassableRun finds the largest contiguous open run", () => {
    const depth = [1, 1, 0.1, 0.1, 1, 1, 1, 0.2, 1];
    const run = widestPassableRun(depth, 0.42);
    assert.equal(run.startCol, 4);
    assert.equal(run.endCol, 7);
    assert.equal(Number(run.widthFraction.toFixed(4)), Number((3 / 9).toFixed(4)));
});

test("widestPassableRun reports zero width when nothing is walkable", () => {
    const run = widestPassableRun([0.1, 0.2, 0.05], 0.42);
    assert.equal(run.widthFraction, 0);
});

test("aggregateClearance weights the worst column heavily", () => {
    const cfg = NavigationConfig.freeSpace;
    const allOpen = aggregateClearance([1, 1, 1, 1], 0, 4, cfg);
    const oneBlocked = aggregateClearance([1, 1, 0, 1], 0, 4, cfg);
    assert.equal(allOpen, 1);
    // Mean is 0.75, min is 0. With minWeight 0.55 the result must be well below
    // the mean: an otherwise-clear sector with one blocked column is not clear.
    assert.ok(oneBlocked < 0.4, `got ${oneBlocked}`);
});

test("an obstacle at your feet closes its columns but leaves the flanks open", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const report = estimator.update(
        [fakeTrack({ b: box(0.36, 0.48, 0.62, 0.80) })],
        corridor.bounds
    );

    assert.ok(report.clearance.center < 0.5, `centre should be blocked, got ${report.clearance.center}`);
    assert.ok(report.clearance.left > 0.9, `left should be open, got ${report.clearance.left}`);
    assert.ok(report.clearance.right > 0.9, `right should be open, got ${report.clearance.right}`);
    assert.equal(report.passable, true, "a gap still exists to one side");
});

test("an object high in the frame does not block the floor", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    // Small, centred, well above the ground band: a distant sign, not an obstacle.
    const report = estimator.update(
        [fakeTrack({ b: box(0.45, 0.20, 0.55, 0.34) })],
        corridor.bounds
    );
    assert.ok(report.clearance.center > 0.9, `got ${report.clearance.center}`);
});

test("a very large object above the ground band still blocks", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    // A bus filling the upper half: its base is high but it is not harmless.
    const report = estimator.update(
        [fakeTrack({ label: "bus", b: box(0.05, 0.02, 0.95, 0.40) })],
        corridor.bounds
    );
    assert.ok(report.clearance.center < 0.85, `got ${report.clearance.center}`);
});

test("a tentative detection only partially reduces free depth", () => {
    const corridor = new CorridorModel();
    const confident = new FreeSpaceEstimator().update(
        [fakeTrack({ b: box(0.36, 0.48, 0.62, 0.80), confidence: 0.95 })],
        corridor.bounds
    );
    const tentative = new FreeSpaceEstimator().update(
        [fakeTrack({ b: box(0.36, 0.48, 0.62, 0.80), confidence: 0.40 })],
        corridor.bounds
    );
    assert.ok(
        tentative.clearance.center > confident.clearance.center,
        "lower confidence must take less depth away"
    );
});

test("both sides blocked leaves no passable gap", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const report = estimator.update(
        [
            fakeTrack({ id: 1, b: box(0.00, 0.46, 0.36, 0.86) }),
            fakeTrack({ id: 2, label: "chair", b: box(0.32, 0.46, 0.68, 0.86) }),
            fakeTrack({ id: 3, label: "suitcase", b: box(0.64, 0.46, 1.00, 0.86) })
        ],
        corridor.bounds
    );
    assert.equal(report.passable, false);
    assert.ok(report.widestGap.widthFraction < NavigationConfig.freeSpace.minPassableWidth);
});

test("segmentation input can veto a column the detector missed", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const n = estimator.columnCount;

    const open = estimator.update([], corridor.bounds);
    assert.ok(open.clearance.center > 0.95, "nothing detected, nothing blocked");

    // A step or kerb across the centre that no COCO class covers.
    const columns = new Float32Array(n).fill(1);
    const from = Math.floor(0.35 * n);
    const to = Math.ceil(0.65 * n);
    for (let i = from; i < to; i += 1) columns[i] = 0.1;

    const refined = estimator.update([], corridor.bounds, { columns });
    assert.ok(refined.clearance.center < 0.4, `got ${refined.clearance.center}`);
});

test("segmentation cannot unblock a detected obstacle", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const columns = new Float32Array(estimator.columnCount).fill(1);
    const report = estimator.update(
        [fakeTrack({ b: box(0.36, 0.48, 0.62, 0.86) })],
        corridor.bounds,
        { columns }
    );
    assert.ok(report.clearance.center < 0.5, "fusion is min(), not max()");
});

/* ------------------------------------------------------------- corridor */

test("the corridor starts conservative and centred", () => {
    const corridor = new CorridorModel();
    assert.equal(corridor.centre, 0.5);
    assert.ok(corridor.left < 0.5 && corridor.right > 0.5);
    assert.equal(
        Number((corridor.right - corridor.left).toFixed(4)),
        Number((NavigationConfig.corridor.baseHalfWidth * 2).toFixed(4))
    );
});

test("the corridor narrows in clutter and widens when open", () => {
    const cluttered = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const tracks = [
        fakeTrack({ id: 1, b: box(0.00, 0.50, 0.24, 0.88) }),
        fakeTrack({ id: 2, b: box(0.30, 0.50, 0.52, 0.88) }),
        fakeTrack({ id: 3, b: box(0.60, 0.50, 0.84, 0.88) })
    ];
    for (let i = 0; i < 40; i += 1) {
        const fs = estimator.update(tracks, cluttered.bounds);
        cluttered.update({ freeSpace: fs, tracks });
    }

    const open = new CorridorModel();
    const openEstimator = new FreeSpaceEstimator();
    for (let i = 0; i < 40; i += 1) {
        const fs = openEstimator.update([], open.bounds);
        open.update({ freeSpace: fs, tracks: [] });
    }

    assert.ok(
        cluttered.halfWidth < open.halfWidth,
        `cluttered ${cluttered.halfWidth} should be narrower than open ${open.halfWidth}`
    );
    assert.ok(open.halfWidth <= NavigationConfig.corridor.maxHalfWidth + 1e-9);
    assert.ok(cluttered.halfWidth >= NavigationConfig.corridor.minHalfWidth - 1e-9);
});

test("the corridor centre does not chase openings", () => {
    // Regression guard. An earlier design steered the corridor centre towards
    // the widest free-space gap. Because the corridor defines the LEFT/CENTRE/
    // RIGHT sector boundaries, that shrank the sector on the open side, inverted
    // the path scores, and produced left/right oscillation. Choosing a side is
    // the decision engine's job; the corridor must stay put.
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();

    // Everything blocked on the left, so the only gap is hard right.
    const tracks = [fakeTrack({ b: box(0.00, 0.46, 0.55, 0.90) })];
    for (let i = 0; i < 200; i += 1) {
        corridor.update({ freeSpace: estimator.update(tracks, corridor.bounds), tracks });
    }

    const max = NavigationConfig.corridor.maxCentreOffset;
    assert.ok(Math.abs(corridor.centre - 0.5) < 1e-6, `centre drifted to ${corridor.centre}`);
    assert.ok(corridor.centre <= 0.5 + max + 1e-6, "and stays inside the hard clamp");
});

test("the corridor stays symmetric under a persistently one-sided scene", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const tracks = [fakeTrack({ b: box(0.00, 0.46, 0.48, 0.90) })];

    const widths = [];
    for (let i = 0; i < 120; i += 1) {
        corridor.update({ freeSpace: estimator.update(tracks, corridor.bounds), tracks });
        widths.push(corridor.right - corridor.left);
        assert.equal(
            Number((corridor.centre).toFixed(6)),
            0.5,
            `centre moved at iteration ${i}`
        );
    }

    // The width may adapt, but it must converge rather than drift.
    const lateSpread = Math.max(...widths.slice(-20)) - Math.min(...widths.slice(-20));
    assert.ok(lateSpread < 0.01, `corridor width never settled, spread ${lateSpread}`);
});

test("corridor evaluate separates path overlap from ground relevance", () => {
    const corridor = new CorridorModel();

    const atFeet = corridor.evaluate(box(0.44, 0.55, 0.56, 0.90));
    assert.equal(atFeet.inCorridor, true);
    assert.ok(atFeet.groundRelevance > 0.9, `got ${atFeet.groundRelevance}`);

    const farAhead = corridor.evaluate(box(0.44, 0.30, 0.56, 0.40));
    assert.ok(farAhead.groundRelevance < 0.15, `got ${farAhead.groundRelevance}`);

    const beside = corridor.evaluate(box(0.02, 0.55, 0.18, 0.90));
    assert.equal(beside.inCorridor, false, "beside the path is not in the path");
    assert.equal(beside.sector, "LEFT");
});

test("corridor coverage and object overlap answer different questions", () => {
    const corridor = new CorridorModel();

    // A narrow pole dead centre: entirely inside the corridor (overlapRatio 1)
    // but covering only a fraction of it.
    const pole = corridor.evaluate(box(0.48, 0.50, 0.52, 0.88));
    assert.equal(Number(pole.overlapRatio.toFixed(3)), 1);
    assert.ok(pole.corridorCoverage < 0.2, `got ${pole.corridorCoverage}`);

    // A wide bench crossing the frame: only part of it is in the corridor, but
    // it covers the whole corridor.
    const bench = corridor.evaluate(box(0.00, 0.55, 1.00, 0.88));
    assert.ok(bench.overlapRatio < 0.5, `got ${bench.overlapRatio}`);
    assert.equal(Number(bench.corridorCoverage.toFixed(3)), 1);
});

test("resetting the corridor restores the conservative default width", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();

    // Drive the width away from its default with a very open scene.
    for (let i = 0; i < 120; i += 1) {
        corridor.update({ freeSpace: estimator.update([], corridor.bounds), tracks: [] });
    }
    const adapted = corridor.halfWidth;
    assert.notEqual(
        Number(adapted.toFixed(4)),
        Number(NavigationConfig.corridor.baseHalfWidth.toFixed(4)),
        "the width should have adapted before we test the reset"
    );

    // A turn resets the corridor: stale geometry must not persist into a new view.
    for (let i = 0; i < 6; i += 1) corridor.update({ reset: true });
    assert.ok(
        Math.abs(corridor.halfWidth - NavigationConfig.corridor.baseHalfWidth)
        < Math.abs(adapted - NavigationConfig.corridor.baseHalfWidth),
        `reset did not move ${corridor.halfWidth} back towards ${NavigationConfig.corridor.baseHalfWidth}`
    );
    assert.equal(Number(corridor.centre.toFixed(6)), 0.5);
});
