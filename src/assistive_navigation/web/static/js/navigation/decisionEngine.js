/**
 * Navigation decision engine.
 *
 * Turns path scores and risk into exactly one action, and - critically - keeps
 * that action stable.
 *
 * The safety hierarchy is absolute (requirement 70):
 *     STOP  >  AVOID  >  CAUTION  >  CONTINUE
 * and object description never competes with collision risk.
 *
 * Hysteresis is the single most important behaviour here. A direction is held
 * for a minimum dwell time and only changes when a *materially* better option
 * has agreed with itself for several frames. Emergency STOP bypasses all of it.
 *
 * Pure, with time passed in. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { Sector } from "../perception/freeSpace.js";

export const Action = Object.freeze({
    STOP: "STOP",
    MOVE_LEFT: "MOVE_LEFT",
    MOVE_SLIGHTLY_LEFT: "MOVE_SLIGHTLY_LEFT",
    MOVE_RIGHT: "MOVE_RIGHT",
    MOVE_SLIGHTLY_RIGHT: "MOVE_SLIGHTLY_RIGHT",
    CAUTION: "CAUTION",
    CONTINUE: "CONTINUE"
});

export const Urgency = Object.freeze({
    CRITICAL: "CRITICAL",
    HIGH: "HIGH",
    MEDIUM: "MEDIUM",
    LOW: "LOW"
});

const DIRECTIONAL = new Set([
    Action.MOVE_LEFT,
    Action.MOVE_SLIGHTLY_LEFT,
    Action.MOVE_RIGHT,
    Action.MOVE_SLIGHTLY_RIGHT
]);

function isLeft(action) {
    return action === Action.MOVE_LEFT || action === Action.MOVE_SLIGHTLY_LEFT;
}

function isRight(action) {
    return action === Action.MOVE_RIGHT || action === Action.MOVE_SLIGHTLY_RIGHT;
}

function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

export class NavigationDecision {
    constructor({
        action = Action.CONTINUE,
        urgency = Urgency.LOW,
        reason = "path_clear",
        confidence = 1,
        primaryObstacle = null,
        pathScores = null,
        emergency = false,
        held = false,
        timestamp = 0
    } = {}) {
        this.action = action;
        this.urgency = urgency;
        this.reason = reason;
        this.confidence = confidence;
        this.primaryObstacle = primaryObstacle;
        this.pathScores = pathScores;
        this.emergency = emergency;
        this.held = held;
        this.timestamp = timestamp;
    }

    get isDirectional() {
        return DIRECTIONAL.has(this.action);
    }

    /**
     * Coarse fingerprint of the situation, used by the speech manager to decide
     * whether anything has *materially* changed (requirement 35). Quantised so
     * that jitter does not produce a new signature every frame.
     */
    signature() {
        // Categorical only. Risk magnitude is deliberately excluded: bucketing a
        // continuous value means a one-point wobble across a bucket edge reads as
        // a brand-new situation and unmutes speech. Magnitude changes are handled
        // separately by the speech policy's `materialRiskDelta` gate, which has
        // no boundary sensitivity.
        const label = this.primaryObstacle?.label || "none";
        const band = this.primaryObstacle?.distanceCategory || "none";
        return `${this.action}|${label}|${band}`;
    }
}

export class DecisionEngine {
    constructor(cfg = NavigationConfig) {
        this.cfg = cfg;
        this.current = new NavigationDecision({ timestamp: 0 });
        this.currentSince = 0;
        this._candidate = null;
        this._candidateFrames = 0;
        this.switchCount = 0;
        this.suppressedSwitches = 0;
        this.clearSince = 0;
        this.wasObstructed = false;
    }

    /**
     * @param {object} input
     * @param {Array} input.tracks scored tracks
     * @param {Array} input.threats risk-sorted tracks above the notice threshold
     * @param {Array} input.emergent emergency-channel tracks
     * @param {object} input.pathScores PathScoreEngine result
     * @param {object} input.freeSpace free-space report
     * @param {object} input.corridor CorridorModel instance
     * @param {number} input.nowMs
     * @param {number} input.qualityScore 0..1
     * @param {boolean} input.qualityAllowsDirection
     * @param {boolean} input.qualityRequiresStop
     * @param {number} [input.degradationLevel] 1 = full pipeline
     * @returns {NavigationDecision}
     */
    evaluate(input) {
        const proposal = this._propose(input);
        return this._applyHysteresis(proposal, input.nowMs);
    }

    /* ---------------------------------------------------------- proposal */

    _propose(input) {
        const {
            threats = [],
            emergent = [],
            pathScores,
            nowMs,
            qualityScore = 1,
            qualityAllowsDirection = true,
            qualityRequiresStop = false,
            freeSpace
        } = input;

        const riskCfg = this.cfg.risk;
        const scoreCfg = this.cfg.pathScore;

        /* --- 0. Perception unusable. Safety outranks everything else. ------ */

        if (qualityRequiresStop) {
            return new NavigationDecision({
                action: Action.STOP,
                urgency: Urgency.HIGH,
                reason: "perception_unreliable",
                confidence: 0.9,
                pathScores,
                emergency: true,
                timestamp: nowMs
            });
        }

        const { left, center, right } = pathScores;
        const primary = threats.length > 0 ? threats[0] : null;
        const emergentPrimary = emergent.length > 0
            ? emergent.slice().sort((a, b) => b.risk - a.risk)[0]
            : null;

        const leftUsable = left.passable && left.adjustedScore >= scoreCfg.blockedThreshold
            && (!left.worst || left.worst.risk < riskCfg.criticalThreshold);
        const rightUsable = right.passable && right.adjustedScore >= scoreCfg.blockedThreshold
            && (!right.worst || right.worst.risk < riskCfg.criticalThreshold);

        /* --- 1. Emergency STOP conditions -------------------------------- */

        const centreCritical = center.risk >= riskCfg.criticalThreshold
            || (primary && primary.risk >= riskCfg.criticalThreshold && primary.corridorEval?.centreBlocked);

        // Both sides gone and the middle is critical: there is nowhere to go.
        if (centreCritical && !leftUsable && !rightUsable) {
            return new NavigationDecision({
                action: Action.STOP,
                urgency: Urgency.CRITICAL,
                reason: "blocked_all_directions",
                confidence: 0.97,
                primaryObstacle: emergentPrimary || primary,
                pathScores,
                emergency: true,
                timestamp: nowMs
            });
        }

        // Something is already at arm's length in the path. At this range a
        // sidestep is not a safe instruction - you cannot reliably clear an
        // obstacle you are about to touch - so stop first.
        const veryNearThreat = (emergentPrimary || primary);
        if (
            veryNearThreat
            && veryNearThreat.distanceCategory === "VERY_NEAR"
            && veryNearThreat.pathOverlap >= 0.35
            && veryNearThreat.risk >= riskCfg.criticalThreshold
        ) {
            return new NavigationDecision({
                action: Action.STOP,
                urgency: Urgency.CRITICAL,
                reason: "obstacle_very_near",
                confidence: 0.96,
                primaryObstacle: veryNearThreat,
                pathScores,
                emergency: true,
                timestamp: nowMs
            });
        }

        // Fast closer on a collision course.
        if (
            veryNearThreat
            && veryNearThreat.rapidlyApproaching
            && veryNearThreat.pathOverlap >= 0.25
            && veryNearThreat.risk >= riskCfg.avoidThreshold
        ) {
            return new NavigationDecision({
                action: Action.STOP,
                urgency: Urgency.CRITICAL,
                reason: "rapid_approach",
                confidence: 0.94,
                primaryObstacle: veryNearThreat,
                pathScores,
                emergency: true,
                timestamp: nowMs
            });
        }

        // Nothing walkable anywhere in the frame.
        if (freeSpace && !freeSpace.passable && center.risk >= riskCfg.avoidThreshold) {
            return new NavigationDecision({
                action: Action.STOP,
                urgency: Urgency.HIGH,
                reason: "no_passable_gap",
                confidence: 0.90,
                primaryObstacle: primary,
                pathScores,
                emergency: true,
                timestamp: nowMs
            });
        }

        /* --- 2. Straight ahead is fine ----------------------------------- */

        const centreSafe = center.adjustedScore >= scoreCfg.safeThreshold
            && center.passable
            && center.risk < riskCfg.avoidThreshold;

        if (centreSafe) {
            return new NavigationDecision({
                action: Action.CONTINUE,
                urgency: Urgency.LOW,
                reason: primary ? "path_clear_ahead" : "path_clear",
                confidence: clamp(0.6 + 0.4 * qualityScore, 0, 1),
                primaryObstacle: null,
                pathScores,
                timestamp: nowMs
            });
        }

        /* --- 3. Choose the safest available side ------------------------- */

        // Only sides that are genuinely usable are eligible. This is the check
        // requirement 28 insists on: never recommend a direction that itself
        // contains a high-risk obstacle.
        const options = [];
        if (leftUsable) options.push(left);
        if (rightUsable) options.push(right);

        /**
         * Tie-break rationale.
         *
         * Two sides scoring within a few points of each other is a coin flip, and
         * a coin flip is the thing that produces LEFT/RIGHT/LEFT flapping. So
         * when the scores are effectively equal we decide on physical evidence
         * instead: first the wider walkable gap, then whichever side the widest
         * opening in the whole frame actually sits on. That is deterministic,
         * explainable, and stable frame to frame.
         */
        const TIE_EPSILON = 6;
        const gapCentre = freeSpace?.bestGapCentre ?? 0.5;
        options.sort((a, b) => {
            const delta = b.adjustedScore - a.adjustedScore;
            if (Math.abs(delta) >= TIE_EPSILON) return delta;
            const widthDelta = b.passableWidth - a.passableWidth;
            if (Math.abs(widthDelta) > 0.02) return widthDelta;
            const aFavoured = (a.sector === Sector.LEFT) === (gapCentre < 0.5);
            const bFavoured = (b.sector === Sector.LEFT) === (gapCentre < 0.5);
            if (aFavoured !== bFavoured) return aFavoured ? -1 : 1;
            return delta;
        });

        const best = options[0] || null;

        if (!best) {
            // Centre is not safe and neither side is usable.
            const severe = center.risk >= riskCfg.avoidThreshold || (primary && primary.risk >= riskCfg.avoidThreshold);
            return new NavigationDecision({
                action: severe ? Action.STOP : Action.CAUTION,
                urgency: severe ? Urgency.HIGH : Urgency.MEDIUM,
                reason: severe ? "blocked_both_sides" : "narrow_path",
                confidence: 0.85,
                primaryObstacle: primary,
                pathScores,
                emergency: severe,
                timestamp: nowMs
            });
        }

        const margin = best.adjustedScore - center.adjustedScore;
        const runnerUp = options[1] || null;
        const sideSeparation = runnerUp ? best.adjustedScore - runnerUp.adjustedScore : best.adjustedScore;

        /**
         * Decision confidence blends:
         *  - how much better the chosen side is than going straight,
         *  - how much better it is than the other side (a coin-flip is low
         *    confidence and should produce caution, not a confident sidestep),
         *  - perception quality,
         *  - how well-established the obstacle we are avoiding is.
         */
        const marginTerm = clamp(margin / 30, 0, 1);
        const separationTerm = clamp(sideSeparation / 25, 0, 1);
        const stabilityTerm = primary ? clamp(primary.stability, 0, 1) : 0.7;
        const confidence = clamp(
            0.25 * marginTerm + 0.25 * separationTerm + 0.30 * qualityScore + 0.20 * stabilityTerm,
            0,
            1
        );

        // Not confident enough to name a side, or perception is too poor to be
        // trusted with a directional instruction (requirements 40, 69).
        if (confidence < this.cfg.hysteresis.minDirectionConfidence || !qualityAllowsDirection) {
            return new NavigationDecision({
                action: Action.CAUTION,
                urgency: center.risk >= riskCfg.avoidThreshold ? Urgency.HIGH : Urgency.MEDIUM,
                reason: qualityAllowsDirection ? "direction_uncertain" : "low_visibility_direction_withheld",
                confidence,
                primaryObstacle: primary,
                pathScores,
                timestamp: nowMs
            });
        }

        const goLeft = best.sector === Sector.LEFT;
        // A large move is warranted when the middle is badly blocked and the
        // chosen side is convincingly open.
        const large = center.adjustedScore < scoreCfg.blockedThreshold && best.adjustedScore >= 68;
        const action = goLeft
            ? (large ? Action.MOVE_LEFT : Action.MOVE_SLIGHTLY_LEFT)
            : (large ? Action.MOVE_RIGHT : Action.MOVE_SLIGHTLY_RIGHT);

        const urgency = center.risk >= riskCfg.avoidThreshold ? Urgency.HIGH : Urgency.MEDIUM;

        return new NavigationDecision({
            action,
            urgency,
            reason: goLeft ? "avoid_left" : "avoid_right",
            confidence,
            primaryObstacle: primary,
            pathScores,
            timestamp: nowMs
        });
    }

    /* -------------------------------------------------------- hysteresis */

    _applyHysteresis(proposal, nowMs) {
        const cfg = this.cfg.hysteresis;
        const held = this.current;

        // Emergencies are never held back and always reset the dwell timer.
        if (proposal.emergency || proposal.urgency === Urgency.CRITICAL || proposal.action === Action.STOP) {
            return this._commit(proposal, nowMs);
        }

        // Nothing held yet, or the same action: just refresh.
        if (held.action === proposal.action) {
            this._candidate = null;
            this._candidateFrames = 0;
            // Keep the freshest supporting evidence, but preserve the original
            // commit time so dwell accounting stays honest.
            proposal.held = true;
            this.current = proposal;
            return proposal;
        }

        const dwell = nowMs - this.currentSince;

        // Returning to CONTINUE after a manoeuvre needs its own, shorter dwell:
        // we want to stop nagging quickly once the way is clear, but not flicker.
        if (proposal.action === Action.CONTINUE) {
            if (dwell < cfg.clearHoldMs) {
                this.suppressedSwitches += 1;
                return this._reHold(nowMs);
            }
            return this._commit(proposal, nowMs);
        }

        if (dwell < cfg.holdMs && held.action !== Action.CONTINUE) {
            // Within the dwell window. Only a materially better option gets in.
            if (!this._beatsHeldByMargin(proposal, held, cfg.switchMarginPoints)) {
                this.suppressedSwitches += 1;
                return this._reHold(nowMs);
            }
        }

        // Require the new option to agree with itself for a few frames. This is
        // what kills LEFT/RIGHT/LEFT flapping when two sectors score similarly.
        if (this._candidate === proposal.action) {
            this._candidateFrames += 1;
        } else {
            this._candidate = proposal.action;
            this._candidateFrames = 1;
        }

        const opposite = (isLeft(held.action) && isRight(proposal.action))
            || (isRight(held.action) && isLeft(proposal.action));
        const requiredFrames = opposite ? cfg.confirmFrames + 1 : cfg.confirmFrames;

        if (this._candidateFrames < requiredFrames) {
            this.suppressedSwitches += 1;
            return this._reHold(nowMs);
        }

        return this._commit(proposal, nowMs);
    }

    /**
     * Does the proposal's target sector beat the held one by enough to justify
     * changing a spoken instruction the user may already be acting on?
     */
    _beatsHeldByMargin(proposal, held, marginPoints) {
        const scores = proposal.pathScores;
        if (!scores) return false;

        const scoreFor = (action) => {
            if (isLeft(action)) return scores.left.adjustedScore;
            if (isRight(action)) return scores.right.adjustedScore;
            return scores.center.adjustedScore;
        };

        if (!DIRECTIONAL.has(held.action) && held.action !== Action.CONTINUE) {
            // Held CAUTION: any confident directional advice is an improvement.
            return proposal.isDirectional;
        }

        return scoreFor(proposal.action) - scoreFor(held.action) >= marginPoints;
    }

    _reHold(nowMs) {
        this.current.held = true;
        this.current.heldSinceMs = this.currentSince;
        this.current.heldForMs = nowMs - this.currentSince;
        return this.current;
    }

    _commit(decision, nowMs) {
        if (decision.action !== this.current.action) this.switchCount += 1;

        this.current = decision;
        this.currentSince = nowMs;
        this._candidate = null;
        this._candidateFrames = 0;

        // Path-clear bookkeeping for the speech manager (requirement 38).
        if (decision.action === Action.CONTINUE) {
            if (this.wasObstructed && this.clearSince === 0) this.clearSince = nowMs;
        } else {
            this.wasObstructed = true;
            this.clearSince = 0;
        }

        return decision;
    }

    /**
     * True when the path has been continuously clear long enough, after a
     * period of obstruction, to be worth announcing once.
     */
    shouldAnnounceClear(nowMs) {
        if (!this.wasObstructed) return false;
        if (this.current.action !== Action.CONTINUE) return false;
        if (this.clearSince === 0) return false;
        return nowMs - this.clearSince >= this.cfg.pathClear.stabilityMs;
    }

    /** Call after announcing, so it is announced once and not again. */
    markClearAnnounced() {
        this.wasObstructed = false;
        this.clearSince = 0;
    }

    reset() {
        this.current = new NavigationDecision({ timestamp: 0 });
        this.currentSince = 0;
        this._candidate = null;
        this._candidateFrames = 0;
        this.clearSince = 0;
        this.wasObstructed = false;
    }

    metrics() {
        return {
            action: this.current.action,
            urgency: this.current.urgency,
            reason: this.current.reason,
            confidence: Number(this.current.confidence.toFixed(2)),
            heldForMs: this.current.heldForMs || 0,
            switches: this.switchCount,
            suppressedSwitches: this.suppressedSwitches,
            candidate: this._candidate,
            candidateFrames: this._candidateFrames
        };
    }
}

export default DecisionEngine;
