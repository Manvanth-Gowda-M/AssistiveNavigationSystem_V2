/**
 * Risk estimation and path scoring.
 *
 * The property that matters most here is *monotonicity*: closer, more central,
 * faster-closing and more confident must never produce a lower risk. If any of
 * those inverts, guidance becomes incoherent in a way that is very hard to spot
 * by walking around.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    RiskEngine,
    convergenceFactor,
    estimateProximity,
    timeToContact
} from "../../src/assistive_navigation/web/static/js/navigation/riskEngine.js";
import {
    PathScoreEngine,
    bandRisk,
    scoreSector
} from "../../src/assistive_navigation/web/static/js/navigation/pathScoring.js";
import { CorridorModel } from "../../src/assistive_navigation/web/static/js/navigation/corridorModel.js";
import { FreeSpaceEstimator } from "../../src/assistive_navigation/web/static/js/perception/freeSpace.js";
import { NavigationConfig } from "../../src/assistive_navigation/web/static/js/config/navigationConfig.js";
import { box } from "./helpers.mjs";

const RISK = NavigationConfig.risk;

/**
 * Track-shaped fixture. Every field the risk engine reads is explicit, so a test
 * failure points at a specific input rather than at tracker behaviour.
 */
function fakeTrack({
    id = 1,
    label = "chair",
    b,
    confidence = 0.9,
    stability = 1,
    visibility = 1,
    velocityX = 0,
    areaGrowthRate = 0,
    emergent = false
} = {}) {
    const [x1, y1, x2, y2] = b;
    return {
        id,
        label,
        confidence,
        visibility,
        velocityX,
        velocityY: 0,
        areaGrowthRate,
        emergent,
        box: [...b],
        bottomY: y2,
        centerX: (x1 + x2) / 2,
        centerY: (y1 + y2) / 2,
        area: (x2 - x1) * (y2 - y1),
        stability,
        get approaching() { return this.areaGrowthRate > 0.12; },
        get rapidlyApproaching() { return this.areaGrowthRate > 0.55; }
    };
}

function scoreOne(track, corridor = new CorridorModel()) {
    const engine = new RiskEngine();
    const evaluation = corridor.evaluate(track.box);
    track.corridorEval = evaluation;
    return engine.score(track, evaluation, corridor.bounds);
}

/* ------------------------------------------------------------- proximity */

test("proximity bands follow ground contact", () => {
    assert.equal(estimateProximity(0.95, 0.10).category, "VERY_NEAR");
    assert.equal(estimateProximity(0.78, 0.08).category, "NEAR");
    assert.equal(estimateProximity(0.62, 0.05).category, "MEDIUM");
    assert.equal(estimateProximity(0.30, 0.005).category, "FAR");
});

test("a large apparent size can promote a band without ground contact", () => {
    // Base high in the frame, but occupying a third of it.
    assert.equal(estimateProximity(0.40, 0.33).category, "VERY_NEAR");
});

test("proximity is monotonic in ground contact", () => {
    let previous = -1;
    for (let y = 0.2; y <= 0.99; y += 0.02) {
        const { proximity } = estimateProximity(y, 0.01);
        assert.ok(proximity >= previous - 1e-9, `proximity dropped at bottomY=${y.toFixed(2)}`);
        previous = proximity;
    }
});

test("timeToContact is infinite when nothing is closing", () => {
    assert.equal(timeToContact(0), Infinity);
    assert.equal(timeToContact(-0.4), Infinity);
    assert.equal(timeToContact(1), 2);
    assert.ok(timeToContact(0.2) > timeToContact(0.8), "slower growth means more time");
});

test("convergence recognises movement into and away from the corridor", () => {
    const corridor = new CorridorModel().bounds;
    const movingIn = fakeTrack({ b: box(0.10, 0.50, 0.26, 0.82), velocityX: 0.3 });
    const movingOut = fakeTrack({ b: box(0.10, 0.50, 0.26, 0.82), velocityX: -0.3 });

    assert.ok(convergenceFactor(movingIn, corridor) > 1, "converging raises risk");
    assert.ok(convergenceFactor(movingOut, corridor) < 1, "diverging lowers it");
    const still = fakeTrack({ b: box(0.10, 0.50, 0.26, 0.82), velocityX: 0 });
    assert.equal(convergenceFactor(still, corridor), 1);
});

/* ------------------------------------------------------------------ risk */

test("an obstacle in the path at near range warrants avoidance", () => {
    const risk = scoreOne(fakeTrack({ b: box(0.36, 0.48, 0.62, 0.78), confidence: 0.88 }));
    assert.ok(risk >= RISK.avoidThreshold, `expected >= ${RISK.avoidThreshold}, got ${risk}`);
    assert.ok(risk < RISK.criticalThreshold, `should not be critical yet, got ${risk}`);
});

test("the same obstacle beside the path is not worth mentioning", () => {
    const risk = scoreOne(fakeTrack({ b: box(0.00, 0.48, 0.20, 0.78), confidence: 0.88 }));
    assert.ok(risk < RISK.noticeThreshold, `expected below notice, got ${risk}`);
});

test("a distant obstacle in the path scores low", () => {
    const risk = scoreOne(fakeTrack({ b: box(0.46, 0.30, 0.54, 0.40), confidence: 0.88 }));
    assert.ok(risk < RISK.noticeThreshold, `got ${risk}`);
});

test("a person at arm's length dead centre is critical", () => {
    const risk = scoreOne(fakeTrack({ label: "person", b: box(0.28, 0.18, 0.72, 0.96), confidence: 0.93 }));
    assert.ok(risk >= RISK.criticalThreshold, `expected critical, got ${risk}`);
});

test("risk is monotonic in proximity", () => {
    let previous = -1;
    for (let y2 = 0.50; y2 <= 0.98; y2 += 0.04) {
        const risk = scoreOne(fakeTrack({ b: box(0.40, y2 - 0.30, 0.60, y2) }));
        assert.ok(risk >= previous, `risk fell from ${previous} to ${risk} at y2=${y2.toFixed(2)}`);
        previous = risk;
    }
});

test("risk is monotonic in confidence", () => {
    let previous = -1;
    for (let c = 0.35; c <= 0.99; c += 0.08) {
        const risk = scoreOne(fakeTrack({ b: box(0.40, 0.50, 0.60, 0.80), confidence: c }));
        assert.ok(risk >= previous, `risk fell at confidence ${c.toFixed(2)}`);
        previous = risk;
    }
});

test("approach raises risk and receding lowers it", () => {
    const baseBox = box(0.40, 0.50, 0.60, 0.78);
    const still = scoreOne(fakeTrack({ b: baseBox }));
    const closing = scoreOne(fakeTrack({ b: baseBox, areaGrowthRate: 0.25 }));
    const rushing = scoreOne(fakeTrack({ b: baseBox, areaGrowthRate: 0.9 }));
    const receding = scoreOne(fakeTrack({ b: baseBox, areaGrowthRate: -0.4 }));

    assert.ok(closing > still, `${closing} vs ${still}`);
    assert.ok(rushing > closing, `${rushing} vs ${closing}`);
    assert.ok(receding < still, `${receding} vs ${still}`);
});

test("hazard class scales risk for identical geometry", () => {
    const b = box(0.38, 0.46, 0.62, 0.78);
    const bagRisk = scoreOne(fakeTrack({ label: "backpack", b }));
    const chairRisk = scoreOne(fakeTrack({ label: "chair", b }));
    const personRisk = scoreOne(fakeTrack({ label: "person", b }));
    const busRisk = scoreOne(fakeTrack({ label: "bus", b }));

    assert.ok(bagRisk < chairRisk, `${bagRisk} < ${chairRisk}`);
    assert.ok(chairRisk < personRisk, `${chairRisk} < ${personRisk}`);
    assert.ok(personRisk < busRisk, `${personRisk} < ${busRisk}`);
});

test("an unstable track carries less weight than a solid one", () => {
    const b = box(0.40, 0.50, 0.60, 0.80);
    const solid = scoreOne(fakeTrack({ b, stability: 1 }));
    const flickering = scoreOne(fakeTrack({ b, stability: 0.1 }));
    assert.ok(flickering < solid, `${flickering} < ${solid}`);
});

test("a half-cropped object is discounted", () => {
    const b = box(0.40, 0.50, 0.60, 0.80);
    const whole = scoreOne(fakeTrack({ b, visibility: 1 }));
    const clipped = scoreOne(fakeTrack({ b, visibility: 0.3 }));
    assert.ok(clipped < whole, `${clipped} < ${whole}`);
});

test("the emergency channel floors risk above the critical threshold", () => {
    // Deliberately weak evidence apart from the emergent flag.
    const risk = scoreOne(fakeTrack({ b: box(0.40, 0.60, 0.60, 0.72), confidence: 0.6, stability: 0.2, emergent: true }));
    assert.ok(risk > RISK.criticalThreshold, `got ${risk}`);
});

test("risk is bounded to 0..100 and records its factors", () => {
    const track = fakeTrack({ label: "bus", b: box(0.02, 0.05, 0.98, 0.99), confidence: 1, areaGrowthRate: 3 });
    const risk = scoreOne(track);
    assert.ok(risk >= 0 && risk <= 100);
    assert.equal(risk, 100, "the worst case saturates rather than overflowing");
    assert.ok(track.riskFactors.classPriority > 1);
    assert.equal(track.distanceCategory, "VERY_NEAR");
});

/* ----------------------------------------------------------- path scores */

test("bandRisk weights by how much of the band an object covers", () => {
    const full = bandRisk([fakeTrack({ b: box(0.30, 0.5, 0.70, 0.85) })].map((t) => ({ ...t, risk: 80 })), 0.30, 0.70, 20);
    const edge = bandRisk([fakeTrack({ b: box(0.66, 0.5, 0.72, 0.85) })].map((t) => ({ ...t, risk: 80 })), 0.30, 0.70, 20);
    assert.ok(full.risk > edge.risk, `${full.risk} > ${edge.risk}`);
    assert.equal(full.risk, 80, "full coverage keeps the full risk");
});

test("bandRisk ignores objects below the notice threshold", () => {
    const result = bandRisk([{ ...fakeTrack({ b: box(0.3, 0.5, 0.7, 0.85) }), risk: 5 }], 0.30, 0.70, 20);
    assert.equal(result.risk, 0);
    assert.equal(result.worst, null);
});

test("an impassably narrow sector is penalised regardless of clearance", () => {
    const wide = scoreSector({ clearance: 0.9, risk: 10, passableWidth: 0.3 });
    const narrow = scoreSector({ clearance: 0.9, risk: 10, passableWidth: 0.05 });
    assert.equal(wide.passable, true);
    assert.equal(narrow.passable, false);
    assert.ok(narrow.score < wide.score * 0.4, `${narrow.score} vs ${wide.score}`);
});

test("free space outweighs risk in the sector score", () => {
    // Same total "badness" distributed differently: the one with less free space
    // must score lower, because free space carries the larger weight.
    const lowFreeSpace = scoreSector({ clearance: 0.2, risk: 0, passableWidth: 0.3 });
    const highRisk = scoreSector({ clearance: 1.0, risk: 80, passableWidth: 0.3 });
    assert.ok(lowFreeSpace.score < highRisk.score, `${lowFreeSpace.score} < ${highRisk.score}`);
});

test("path scoring prefers the genuinely open side, not the emptier-looking one", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const engine = new RiskEngine();
    const scorer = new PathScoreEngine();

    // Centre blocked by a chair; the right flank holds a bicycle, the left is clear.
    const tracks = [
        fakeTrack({ id: 1, label: "chair", b: box(0.36, 0.48, 0.62, 0.80) }),
        fakeTrack({ id: 2, label: "bicycle", b: box(0.70, 0.48, 1.00, 0.82) })
    ];

    const freeSpace = estimator.update(tracks, corridor.bounds);
    engine.scoreAll(tracks, corridor);
    const scores = scorer.evaluate(tracks, freeSpace, corridor);

    assert.ok(scores.left.score > scores.right.score, `left ${scores.left.score} vs right ${scores.right.score}`);
    assert.ok(scores.left.score > scores.center.score, "and better than straight ahead");
    assert.equal(scores.best, "LEFT");
});

test("a lateral move carries a cost, so a marginal advantage is not enough", () => {
    const corridor = new CorridorModel();
    const estimator = new FreeSpaceEstimator();
    const scorer = new PathScoreEngine();
    const freeSpace = estimator.update([], corridor.bounds);
    const scores = scorer.evaluate([], freeSpace, corridor);

    assert.equal(scores.center.adjustedScore, scores.center.score, "going straight is free");
    assert.ok(scores.left.adjustedScore < scores.left.score, "sidestepping is not");
    assert.equal(scores.best, "CENTER", "with everything clear, straight ahead wins");
});
