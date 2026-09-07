/**
 * Path scoring: left / centre / right, 0-100.
 *
 * This is the module that makes multi-object reasoning work. The pre-upgrade
 * engine looked at the single highest-risk obstacle, asked which flank had
 * "more free space", and steered there - which is how you get told to move into
 * a bicycle because the bicycle happened to be smaller than the chair in front
 * of you.
 *
 * Here every sector is scored independently from two things:
 *   - free space actually available in that sector (weighted higher), and
 *   - the worst obstacle risk present in that sector.
 *
 * A sector narrower than a shoulder width is penalised hard regardless of how
 * empty it looks, because "clear" and "walkable" are different claims.
 *
 * Pure. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { Sector } from "../perception/freeSpace.js";

function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/**
 * Horizontal overlap fraction of a box with a [lo,hi] band, relative to the
 * band width.
 */
function bandCoverage(box, lo, hi) {
    const span = hi - lo;
    if (span <= 1e-6) return 0;
    const overlap = Math.max(0, Math.min(box[2], hi) - Math.max(box[0], lo));
    return overlap / span;
}

/**
 * Worst risk present in a horizontal band.
 *
 * Weighted by how much of the band the object covers, so an object clipping the
 * edge of a sector does not condemn the whole sector, and by ground relevance so
 * a high/distant object does not either.
 *
 * @returns {{risk:number, worst:object|null}}
 */
export function bandRisk(tracks, lo, hi, noticeThreshold) {
    let worstRisk = 0;
    let worst = null;

    for (const track of tracks) {
        if (track.risk < noticeThreshold) continue;
        const coverage = bandCoverage(track.box, lo, hi);
        if (coverage <= 0.04) continue;

        // sqrt() so partial coverage still counts substantially: a pole covering
        // 25% of a sector is a real obstruction, not a quarter of one.
        const weighted = track.risk * Math.sqrt(clamp(coverage, 0, 1));
        if (weighted > worstRisk) {
            worstRisk = weighted;
            worst = track;
        }
    }

    return { risk: Math.min(100, Math.round(worstRisk)), worst };
}

/**
 * Score one sector.
 *
 * @returns {{score:number, clearance:number, risk:number, passableWidth:number,
 *            passable:boolean, worst:object|null}}
 */
export function scoreSector({ clearance, risk, passableWidth, cfg = NavigationConfig.pathScore }) {
    const freeTerm = clamp(clearance, 0, 1);
    const riskTerm = 1 - clamp(risk / 100, 0, 1);

    let score = 100 * (cfg.freeSpaceWeight * freeTerm + cfg.riskWeight * riskTerm);

    const passable = passableWidth >= NavigationConfig.freeSpace.minPassableWidth;
    if (!passable) {
        // Not a soft nudge: a gap you cannot fit through is not a path.
        score *= cfg.impassablePenalty;
    }

    return {
        score: Math.round(clamp(score, 0, 100)),
        clearance: freeTerm,
        risk,
        passableWidth,
        passable
    };
}

export class PathScoreEngine {
    constructor(cfg = NavigationConfig.pathScore) {
        this.cfg = cfg;
        this.last = null;
    }

    /**
     * @param {Array} tracks scored tracks
     * @param {object} freeSpace free-space report
     * @param {object} corridor CorridorModel instance
     * @returns {{left:object, center:object, right:object, best:string,
     *            bestScore:number, ordered:Array}}
     */
    evaluate(tracks, freeSpace, corridor) {
        const bounds = corridor.bounds;
        const notice = NavigationConfig.risk.noticeThreshold;

        const leftBand = [0, bounds.left];
        const centreBand = [bounds.left, bounds.right];
        const rightBand = [bounds.right, 1];

        const leftRisk = bandRisk(tracks, leftBand[0], leftBand[1], notice);
        const centreRisk = bandRisk(tracks, centreBand[0], centreBand[1], notice);
        const rightRisk = bandRisk(tracks, rightBand[0], rightBand[1], notice);

        const left = {
            sector: Sector.LEFT,
            ...scoreSector({
                clearance: freeSpace.clearance.left,
                risk: leftRisk.risk,
                passableWidth: freeSpace.passableWidth.left,
                cfg: this.cfg
            }),
            worst: leftRisk.worst
        };
        const center = {
            sector: Sector.CENTER,
            ...scoreSector({
                clearance: freeSpace.clearance.center,
                risk: centreRisk.risk,
                passableWidth: freeSpace.passableWidth.center,
                cfg: this.cfg
            }),
            worst: centreRisk.worst
        };
        const right = {
            sector: Sector.RIGHT,
            ...scoreSector({
                clearance: freeSpace.clearance.right,
                risk: rightRisk.risk,
                passableWidth: freeSpace.passableWidth.right,
                cfg: this.cfg
            }),
            worst: rightRisk.worst
        };

        // Deviating from straight ahead costs something. Without this the system
        // would recommend a sidestep for a one-point score advantage, which
        // feels twitchy and is not useful guidance.
        left.adjustedScore = Math.max(0, left.score - this.cfg.lateralMoveCost);
        right.adjustedScore = Math.max(0, right.score - this.cfg.lateralMoveCost);
        center.adjustedScore = center.score;

        const ordered = [center, left, right].slice().sort((a, b) => b.adjustedScore - a.adjustedScore);

        const result = {
            left,
            center,
            right,
            ordered,
            best: ordered[0].sector,
            bestScore: ordered[0].adjustedScore,
            bounds
        };
        this.last = result;
        return result;
    }

    metrics() {
        if (!this.last) return null;
        return {
            left: this.last.left.score,
            center: this.last.center.score,
            right: this.last.right.score,
            best: this.last.best
        };
    }
}

export default PathScoreEngine;
