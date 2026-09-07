/**
 * Collision risk estimation.
 *
 * Produces a 0-100 risk score per track from normalised, independently
 * meaningful factors. Multiplicative rather than additive on purpose: if any
 * single factor says "this cannot hit me" - it is far away, it is outside my
 * path, it is not really there - the product collapses and the object stops
 * driving decisions. An additive model lets a confident, large, irrelevant
 * object accumulate a dangerous-looking score, which is how you end up stopping
 * for a parked car on the far side of the road.
 *
 * Factors, all normalised:
 *   proximity      how close, from ground contact and apparent size
 *   path           how much of my walking corridor it occupies
 *   ground         whether it is on the floor in front of me vs high/distant
 *   confidence     how sure the detector is
 *   stability      how consistently the tracker has held it
 *   visibility     how much of it survived the display crop
 *   approach       is it closing, and how fast
 *   convergence    is it moving into my path or out of it
 *   class priority a bus is not a backpack
 *
 * Pure. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { classPriority } from "../config/classCatalog.js";

/**
 * Normalisation divisor.
 *
 * Chosen so that the worst realistic case (a very near, centred, confident,
 * rapidly-approaching vehicle) saturates at 100, while a merely near, centred,
 * moderately-confident piece of furniture lands just above the avoidance
 * threshold. Changing this rescales every threshold in NavigationConfig.risk,
 * so the unit tests pin the important boundaries.
 */
const RISK_NORMALISER = 1.45;

function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * Approximate distance band.
 *
 * Ground contact (`bottomY`) is the primary cue and apparent area is the
 * secondary one, because ground contact degrades gracefully: a partially
 * occluded object still touches the floor where it touches the floor, whereas
 * its area can halve. Either cue can promote a band - a bus filling the frame
 * without its base visible is still near.
 *
 * @returns {{category:string, proximity:number}}
 */
export function estimateProximity(bottomY, area, bands = NavigationConfig.proximity) {
    for (const key of ["VERY_NEAR", "NEAR", "MEDIUM"]) {
        const band = bands[key];
        if (bottomY >= band.minBottomY || area >= band.minArea) {
            // Interpolate within the band so risk does not jump at boundaries.
            const next = key === "VERY_NEAR" ? null : bands[key === "NEAR" ? "VERY_NEAR" : "NEAR"];
            let proximity = band.proximity;
            if (next) {
                const t = clamp((bottomY - band.minBottomY) / Math.max(1e-6, next.minBottomY - band.minBottomY), 0, 1);
                proximity = band.proximity + (next.proximity - band.proximity) * t;
            }
            return { category: key, proximity };
        }
    }
    const far = bands.FAR;
    // Within FAR, scale down further with height in frame so a speck on the
    // horizon scores near zero rather than a flat 0.22.
    const t = clamp(bottomY / Math.max(1e-6, bands.MEDIUM.minBottomY), 0, 1);
    return { category: "FAR", proximity: far.proximity * (0.35 + 0.65 * t) };
}

/**
 * Rough time-to-contact from relative area growth.
 *
 * Apparent area scales as 1/d^2, so the relative growth rate of area is
 * 2 * (closing speed) / distance, which makes `2 / growthRate` an estimate of
 * seconds to contact. It is crude and it is only used as a risk multiplier, not
 * reported to the user.
 *
 * @returns {number} seconds, or Infinity when not closing
 */
export function timeToContact(areaGrowthRate) {
    if (!(areaGrowthRate > 0.02)) return Infinity;
    return 2 / areaGrowthRate;
}

/**
 * Is the object moving into the corridor or out of it?
 * @returns {number} multiplier
 */
export function convergenceFactor(track, corridor, cfg = NavigationConfig.risk) {
    const centre = corridor.centre;
    const dx = track.centerX - centre;
    const vx = track.velocityX;

    // Not meaningfully moving sideways, or already dead centre.
    if (Math.abs(vx) < 0.04 || Math.abs(dx) < 0.02) return 1;

    // Negative product means the object is moving towards the corridor centre.
    const closing = dx * vx < 0;
    if (closing) {
        const inside = Math.abs(dx) <= corridor.width / 2;
        // Something already in the corridor and still converging is worse than
        // something merely heading towards it.
        return inside ? cfg.convergingFactor : 1 + (cfg.convergingFactor - 1) * 0.6;
    }
    return cfg.divergingFactor;
}

export class RiskEngine {
    constructor(cfg = NavigationConfig.risk) {
        this.cfg = cfg;
    }

    /**
     * Score one track and write the results onto it.
     *
     * Mutating the track is deliberate: downstream stages (path scoring,
     * decision, speech, overlay) all need these values, and copying them into a
     * parallel structure every frame is allocation we do not need.
     *
     * @param {object} track
     * @param {object} corridorEval result of CorridorModel.evaluate()
     * @param {object} corridorBounds
     * @returns {number} risk 0-100
     */
    score(track, corridorEval, corridorBounds) {
        const cfg = this.cfg;

        const { category, proximity } = estimateProximity(track.bottomY, track.area);
        track.distanceCategory = category;
        track.proximity = proximity;

        // Path term: the larger of "how much of the object is in my path" and
        // "how much of my path it covers". A narrow pole dead centre blocks the
        // path even though most of the corridor is technically open beside it;
        // a wide bench crossing the corridor blocks it even if most of the
        // bench is outside.
        const pathTerm = Math.max(corridorEval.overlapRatio, corridorEval.corridorCoverage);
        track.pathOverlap = pathTerm;
        track.groundContact = corridorEval.groundRelevance;

        const pathFactor = 0.25 + 0.75 * clamp(pathTerm, 0, 1);
        const groundFactor = 0.30 + 0.70 * clamp(corridorEval.groundRelevance, 0, 1);
        const confidenceFactor = 0.55 + 0.45 * clamp(track.confidence, 0, 1);
        const stabilityFactor = cfg.minStabilityFactor + (1 - cfg.minStabilityFactor) * clamp(track.stability, 0, 1);
        const visibilityFactor = 0.55 + 0.45 * clamp(track.visibility ?? 1, 0, 1);

        let approachFactor = 1;
        if (track.rapidlyApproaching) approachFactor = cfg.rapidApproachFactor;
        else if (track.approaching) approachFactor = cfg.approachFactor;
        else if (track.areaGrowthRate < -0.12) approachFactor = cfg.recedingFactor;

        const ttc = timeToContact(track.areaGrowthRate);
        track.ttcSeconds = ttc;
        // Imminent contact overrides the coarse approach bands.
        if (ttc < 1.5) approachFactor = Math.max(approachFactor, cfg.rapidApproachFactor);
        else if (ttc < 3) approachFactor = Math.max(approachFactor, cfg.approachFactor);

        const converge = convergenceFactor(track, corridorBounds, cfg);
        const priority = classPriority(track.label);

        const threat = proximity
            * pathFactor
            * groundFactor
            * confidenceFactor
            * stabilityFactor
            * visibilityFactor
            * approachFactor
            * converge
            * priority;

        let risk = Math.round(100 * clamp(threat / RISK_NORMALISER, 0, 1));

        // The emergency channel bypasses tracker confirmation, so it must also
        // bypass the stability and confidence discounts that keep a one-frame
        // detection from dominating. It cannot bypass geometry: `emergent` is
        // only set for large, centred, close or fast-closing objects.
        if (track.emergent) {
            risk = Math.max(risk, cfg.criticalThreshold + 2);
        }

        track.risk = risk;
        track.riskFactors = {
            proximity: Number(proximity.toFixed(3)),
            pathFactor: Number(pathFactor.toFixed(3)),
            groundFactor: Number(groundFactor.toFixed(3)),
            confidenceFactor: Number(confidenceFactor.toFixed(3)),
            stabilityFactor: Number(stabilityFactor.toFixed(3)),
            visibilityFactor: Number(visibilityFactor.toFixed(3)),
            approachFactor: Number(approachFactor.toFixed(3)),
            convergenceFactor: Number(converge.toFixed(3)),
            classPriority: Number(priority.toFixed(2)),
            ttcSeconds: Number.isFinite(ttc) ? Number(ttc.toFixed(2)) : null
        };

        return risk;
    }

    /**
     * Score every track and return the ones that matter, highest risk first.
     *
     * @param {Array} tracks
     * @param {object} corridor CorridorModel instance
     * @returns {{scored:Array, threats:Array, primary:object|null, maxRisk:number}}
     */
    scoreAll(tracks, corridor) {
        const bounds = corridor.bounds;
        const threats = [];
        let maxRisk = 0;

        for (const track of tracks) {
            const evaluation = corridor.evaluate(track.box);
            track.corridorEval = evaluation;
            const risk = this.score(track, evaluation, bounds);
            if (risk > maxRisk) maxRisk = risk;
            if (risk >= this.cfg.noticeThreshold) threats.push(track);
        }

        threats.sort((a, b) => b.risk - a.risk);

        return {
            scored: tracks,
            threats,
            primary: threats.length > 0 ? threats[0] : null,
            maxRisk
        };
    }
}

export default RiskEngine;
