/**
 * Free-space estimation.
 *
 * This is the upgrade that changes the system's character. Bounding boxes answer
 * "what objects exist"; a free-space grid answers "where can I actually put my
 * feet", which is the only question that matters when you are walking.
 *
 * Representation: a 1-D column occupancy grid across the navigation width. For
 * each column we compute a **free depth** in 0..1 - how far forward that column
 * is walkable before something blocks it. A column with an object at the very
 * bottom of the frame (at your feet) has free depth 0; an empty column has 1.
 *
 * Why 1-D rather than a 2-D occupancy map: the decision we need to make is
 * "left, straight, right, or stop". A column grid answers that directly, costs
 * a few hundred operations, and cannot drift out of sync with the tracker. A
 * 2-D map would be more general and would buy us nothing here.
 *
 * An optional segmentation signal can be fused in (see segmentationStage.js).
 * Fusion is `min`: either source may veto a column, neither may unblock one.
 *
 * Pure. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { classPriority } from "../config/classCatalog.js";

export const Sector = Object.freeze({
    LEFT: "LEFT",
    CENTER: "CENTER",
    RIGHT: "RIGHT"
});

/**
 * Map an object's ground-contact row to a free-depth value.
 *
 * `bottomY` near 1.0 means the object touches the bottom of the frame, i.e. it
 * is immediately in front of the user, so remaining depth is 0. `bottomY` at or
 * above the ground-band top means it is far enough away to leave the column
 * essentially open.
 */
export function bottomYToFreeDepth(bottomY, groundBandTop) {
    const span = 1 - groundBandTop;
    if (span <= 0) return 1;
    const depth = (1 - bottomY) / span;
    return depth < 0 ? 0 : (depth > 1 ? 1 : depth);
}

/**
 * Widest contiguous run of walkable columns.
 *
 * @param {Float32Array|number[]} freeDepth
 * @param {number} threshold
 * @returns {{startCol:number, endCol:number, widthFraction:number, centreFraction:number}}
 *          `endCol` is exclusive. Returns width 0 when nothing is walkable.
 */
export function widestPassableRun(freeDepth, threshold) {
    const n = freeDepth.length;
    let bestStart = 0;
    let bestLen = 0;
    let runStart = -1;

    for (let i = 0; i <= n; i += 1) {
        const walkable = i < n && freeDepth[i] >= threshold;
        if (walkable) {
            if (runStart === -1) runStart = i;
        } else if (runStart !== -1) {
            const len = i - runStart;
            if (len > bestLen) {
                bestLen = len;
                bestStart = runStart;
            }
            runStart = -1;
        }
    }

    if (bestLen === 0) {
        return { startCol: 0, endCol: 0, widthFraction: 0, centreFraction: 0.5 };
    }
    return {
        startCol: bestStart,
        endCol: bestStart + bestLen,
        widthFraction: bestLen / n,
        centreFraction: (bestStart + bestLen / 2) / n
    };
}

/** Aggregate a column range into a single clearance value in 0..1. */
export function aggregateClearance(freeDepth, startCol, endCol, cfg) {
    if (endCol <= startCol) return 0;
    let sum = 0;
    let min = 1;
    for (let i = startCol; i < endCol; i += 1) {
        const v = freeDepth[i];
        sum += v;
        if (v < min) min = v;
    }
    const mean = sum / (endCol - startCol);
    return Math.max(0, Math.min(1, cfg.meanWeight * mean + cfg.minWeight * min));
}

export class FreeSpaceEstimator {
    constructor(cfg = NavigationConfig.freeSpace) {
        this.cfg = cfg;
        this.columnCount = cfg.columns;

        /** Free depth per column, 1 = fully open. */
        this.freeDepth = new Float32Array(this.columnCount).fill(1);
        /** Occupancy weight per column, driven by confidence and hazard class. */
        this.occupancy = new Float32Array(this.columnCount);
        /** Track id blocking each column, or -1. Used for explanations. */
        this.blockerId = new Int32Array(this.columnCount).fill(-1);

        this._detectionDepth = new Float32Array(this.columnCount).fill(1);

        this.lastResult = null;
    }

    /**
     * @param {Array} tracks tracker output (navigation coordinates)
     * @param {object} corridor { left, right, centre } normalised bounds
     * @param {{columns?:Float32Array}|null} segmentation optional per-column ground freeness
     * @returns {object} free-space report
     */
    update(tracks, corridor, segmentation = null) {
        const n = this.columnCount;
        const cfg = this.cfg;
        const depth = this._detectionDepth;
        const occupancy = this.occupancy;
        const blockerId = this.blockerId;

        depth.fill(1);
        occupancy.fill(0);
        blockerId.fill(-1);

        for (const track of tracks) {
            // Ignored classes were filtered upstream; this guards the weight.
            const priority = classPriority(track.label);
            if (priority <= 0) continue;

            // Coasted tracks still block: an object does not cease to exist
            // because one frame missed it. Their influence is scaled by the
            // decayed confidence.
            const evidence = Math.max(0, Math.min(1, track.confidence)) * (track.visibility ?? 1);
            if (evidence <= 0.05) continue;

            // Above the ground band an object is either distant or overhead. A
            // very large one still matters (a bus filling the top half is not
            // "far"), so the gate is on size as well as position.
            const groundVisible = track.bottomY >= cfg.groundBandTop;
            if (!groundVisible && track.area < 0.22) continue;

            let objectDepth = bottomYToFreeDepth(track.bottomY, cfg.groundBandTop);

            if (!groundVisible) {
                // No ground contact in frame, so the vertical cue is unusable -
                // it would report "fully open" for a wall two feet away. Fall
                // back to apparent size: something occupying a large fraction of
                // the frame is close, whatever its base is doing.
                const sizeDepth = Math.max(0, Math.min(1, 1 - track.area / 0.45));
                objectDepth = Math.min(objectDepth, sizeDepth);
            }

            // Widen the footprint slightly: a real object is wider than its box
            // at floor level, and clipping past an obstacle by 2 cm is not a
            // manoeuvre we should recommend.
            const margin = 0.015 + 0.05 * (1 - objectDepth);
            const x1 = Math.max(0, track.box[0] - margin);
            const x2 = Math.min(1, track.box[2] + margin);

            const startCol = Math.max(0, Math.floor(x1 * n));
            const endCol = Math.min(n, Math.ceil(x2 * n));

            // Confidence and hazard class modulate how much depth is taken.
            // A tentative detection reduces free depth partially rather than
            // slamming the column shut, which is what makes the grid tolerate
            // detector flicker.
            const strength = Math.min(1, evidence * Math.min(1.4, priority));
            const effectiveDepth = objectDepth + (1 - objectDepth) * (1 - strength);

            for (let c = startCol; c < endCol; c += 1) {
                if (effectiveDepth < depth[c]) {
                    depth[c] = effectiveDepth;
                    blockerId[c] = track.id;
                }
                const occ = (1 - effectiveDepth) * strength;
                if (occ > occupancy[c]) occupancy[c] = occ;
            }
        }

        // Fuse the optional segmentation signal. `min` semantics: segmentation
        // may reveal a step, kerb or wall the detector has no class for, but it
        // may never declare a detected obstacle absent.
        const fused = this.freeDepth;
        if (segmentation?.columns && segmentation.columns.length === n) {
            for (let c = 0; c < n; c += 1) {
                fused[c] = Math.min(depth[c], Math.max(0, Math.min(1, segmentation.columns[c])));
            }
        } else {
            fused.set(depth);
        }

        return this._report(fused, corridor);
    }

    _report(freeDepth, corridor) {
        const n = this.columnCount;
        const cfg = this.cfg;

        const leftEnd = Math.max(1, Math.round(corridor.left * n));
        const rightStart = Math.min(n - 1, Math.round(corridor.right * n));

        const left = aggregateClearance(freeDepth, 0, leftEnd, cfg);
        const center = aggregateClearance(freeDepth, leftEnd, rightStart, cfg);
        const right = aggregateClearance(freeDepth, rightStart, n, cfg);

        const gap = widestPassableRun(freeDepth, cfg.passableDepthThreshold);
        const passable = gap.widthFraction >= cfg.minPassableWidth;

        // Per-sector passable width, so a sector can be "clear on average" but
        // still too narrow to walk through.
        const leftGap = widestPassableRun(freeDepth.subarray(0, leftEnd), cfg.passableDepthThreshold);
        const centerGap = widestPassableRun(freeDepth.subarray(leftEnd, rightStart), cfg.passableDepthThreshold);
        const rightGap = widestPassableRun(freeDepth.subarray(rightStart, n), cfg.passableDepthThreshold);

        const result = {
            columnCount: n,
            freeDepth,
            occupancy: this.occupancy,
            blockerId: this.blockerId,
            clearance: { left, center, right },
            /** Absolute passable width per sector, as a fraction of the frame. */
            passableWidth: {
                left: (leftGap.widthFraction * leftEnd) / n,
                center: (centerGap.widthFraction * (rightStart - leftEnd)) / n,
                right: (rightGap.widthFraction * (n - rightStart)) / n
            },
            /** Widest gap anywhere in the frame. */
            widestGap: gap,
            passable,
            /** Where the widest gap sits, in navigation coordinates. */
            bestGapCentre: gap.centreFraction,
            sectorBounds: { leftEnd, rightStart }
        };

        this.lastResult = result;
        return result;
    }

    /** Which sector a normalised x falls into, given the corridor. */
    static sectorFor(x, corridor) {
        if (x < corridor.left) return Sector.LEFT;
        if (x > corridor.right) return Sector.RIGHT;
        return Sector.CENTER;
    }

    metrics() {
        const r = this.lastResult;
        if (!r) return null;
        return {
            left: Number(r.clearance.left.toFixed(2)),
            center: Number(r.clearance.center.toFixed(2)),
            right: Number(r.clearance.right.toFixed(2)),
            widestGapWidth: Number(r.widestGap.widthFraction.toFixed(2)),
            widestGapCentre: Number(r.widestGap.centreFraction.toFixed(2)),
            passable: r.passable
        };
    }
}
