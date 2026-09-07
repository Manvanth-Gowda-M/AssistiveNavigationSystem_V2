/**
 * Optional free-space refinement stage, with an adaptive on/off controller.
 *
 * Two things live here.
 *
 * 1. `GroundFreenessEstimator` - a classical, model-free ground-plane
 *    free-space estimator. It walks upward from the bottom of the analysis
 *    thumbnail, per column, comparing appearance against a floor model
 *    bootstrapped from the rows nearest the user, and stops at the first strong
 *    discontinuity. That row is where the walkable floor ends.
 *
 *    This exists because the obstacles that hurt most are the ones COCO has no
 *    class for: a kerb, a step, a pallet, a wall, a closed door, a puddle edge.
 *    Requirement 64/65 - geometry beats identity. It costs about 3,000
 *    integer operations on a 64x48 thumbnail, which is roughly 0.05 ms, so it
 *    can run on every inference without a benchmark argument.
 *
 *    It replaces the pre-upgrade `barrierDetector.js`, which declared a
 *    frame-filling "wall" at 0.85 confidence whenever the centre of the image
 *    was low-texture. That fired on plain floors, roads and sky, and every one
 *    of those became a false STOP.
 *
 * 2. `AdaptiveSegmentationController` - the load governor. It measures what the
 *    stage costs, runs it only when it can change a decision, and switches it
 *    off entirely if it starts hurting. A neural segmenter can be plugged in
 *    through the same interface and is subject to the same governor
 *    (requirements 21, 22).
 *
 * Pure computation. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { bottomYToFreeDepth } from "./freeSpace.js";

/** Reasons the stage ran or did not run, for the dashboard. */
export const SegmentationTrigger = Object.freeze({
    OBSTACLE_PRESENT: "obstacle-present",
    DIRECTION_UNCERTAIN: "direction-uncertain",
    SCENE_CHANGED: "scene-changed",
    PERIODIC: "periodic",
    SKIPPED_BUDGET: "skipped-budget",
    DISABLED: "disabled"
});

export class GroundFreenessEstimator {
    /**
     * @param {{columns?:number, groundBandTop?:number,
     *          edgeThreshold?:number, appearanceThreshold?:number}} [opts]
     */
    constructor({
        columns = NavigationConfig.freeSpace.columns,
        groundBandTop = NavigationConfig.freeSpace.groundBandTop,
        edgeThreshold = 26,
        appearanceThreshold = 34,
        /**
         * Consecutive deviating rows required before a discontinuity counts as an
         * obstacle rather than a floor marking. On a 48-row analysis thumbnail, 3
         * rows is roughly 6% of frame height - taller than any painted line or
         * grout seam, shorter than any real object.
         */
        minObstacleRows = 3
    } = {}) {
        this.columns = columns;
        this.groundBandTop = groundBandTop;
        this.edgeThreshold = edgeThreshold;
        this.appearanceThreshold = appearanceThreshold;
        this.minObstacleRows = minObstacleRows;

        this.out = new Float32Array(columns).fill(1);
        /** Row index (in output-column space) where each column became blocked. */
        this.blockRow = new Int32Array(columns).fill(-1);
        /**
         * Per-thumbnail-column block row. Sized to the thumbnail width, not the
         * output column count - those are different lengths, and conflating them
         * silently drops out-of-range writes on a typed array.
         */
        this._pixelBlockRow = new Int32Array(64);
        this.lastCostMs = 0;
    }

    _ensurePixelScratch(width) {
        if (this._pixelBlockRow.length < width) {
            this._pixelBlockRow = new Int32Array(width);
        }
    }

    /**
     * @param {Uint8Array} luma
     * @param {number} width
     * @param {number} height
     * @returns {{columns:Float32Array, blockRow:Int32Array, floorMean:number, costMs:number}}
     */
    estimate(luma, width, height) {
        const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
        const nCols = this.columns;
        const out = this.out;
        const blockRow = this.blockRow;
        out.fill(1);
        blockRow.fill(-1);

        if (!luma || width < 4 || height < 6) {
            this.lastCostMs = 0;
            return { columns: out, blockRow, floorMean: 0, costMs: 0 };
        }

        // Bootstrap the floor model from the bottom 12% of rows. Those rows are
        // within a stride of the user; if they are not floor, the user is
        // already against something and the detector path will say so.
        const bootstrapRows = Math.max(2, Math.round(height * 0.12));
        let floorSum = 0;
        let floorCount = 0;
        for (let y = height - bootstrapRows; y < height; y += 1) {
            const row = y * width;
            for (let x = 0; x < width; x += 1) {
                floorSum += luma[row + x];
                floorCount += 1;
            }
        }
        const floorMean = floorCount > 0 ? floorSum / floorCount : 0;

        // Per thumbnail column, find the first row (walking up) that stops
        // looking like floor.
        this._ensurePixelScratch(width);
        const colBlockRow = this._pixelBlockRow;
        for (let x = 0; x < width; x += 1) colBlockRow[x] = -1;

        for (let x = 0; x < width; x += 1) {
            // Local floor reference: blend the global model with this column's
            // own bottom rows, so a shadow down one side does not read as a wall.
            let localSum = 0;
            for (let y = height - bootstrapRows; y < height; y += 1) localSum += luma[y * width + x];
            const localFloor = 0.5 * floorMean + 0.5 * (localSum / bootstrapRows);

            let y = height - bootstrapRows - 1;
            while (y >= 1) {
                const deviates = Math.abs(luma[y * width + x] - localFloor) >= this.appearanceThreshold
                    || Math.abs(luma[y * width + x] - luma[(y + 1) * width + x]) >= this.edgeThreshold;

                if (!deviates) {
                    y -= 1;
                    continue;
                }

                /**
                 * A deviation is only an obstacle if it *persists* upward.
                 *
                 * This distinguishes the two cases that look identical for a single
                 * row. An obstacle occludes everything above its base, so the
                 * deviation continues. A floor marking - tiling grout, a painted
                 * line, an expansion joint, a shadow edge - is a thin band with
                 * floor visible again above it.
                 *
                 * Without this the estimator reports "floor ends here" at the first
                 * pavement joint, which caps free depth across the entire frame and
                 * makes an open path look partly blocked. Measured on a synthetic
                 * floor with scrolling seams: clearance 0.75 before, 1.0 after.
                 */
                let run = 0;
                let probe = y;
                while (probe >= 1 && run < this.minObstacleRows) {
                    if (Math.abs(luma[probe * width + x] - localFloor) < this.appearanceThreshold) break;
                    run += 1;
                    probe -= 1;
                }

                if (run >= this.minObstacleRows) {
                    colBlockRow[x] = y;
                    break;
                }

                // A thin band: skip past it and keep looking for real structure.
                y = probe - 1;
            }
        }

        // Reduce thumbnail columns to output columns, taking the *lowest* block
        // row (nearest obstacle) in each bucket.
        const perBucket = width / nCols;
        for (let c = 0; c < nCols; c += 1) {
            const xStart = Math.floor(c * perBucket);
            const xEnd = Math.max(xStart + 1, Math.floor((c + 1) * perBucket));

            let nearest = -1;
            for (let x = xStart; x < xEnd && x < width; x += 1) {
                const row = colBlockRow[x];
                if (row === -1) continue;
                if (nearest === -1 || row > nearest) nearest = row;
            }

            if (nearest === -1) {
                out[c] = 1;
                blockRow[c] = -1;
            } else {
                blockRow[c] = nearest;
                // The obstacle's floor-contact row behaves exactly like a
                // bounding box's bottom edge, so reuse the same mapping.
                out[c] = bottomYToFreeDepth((nearest + 1) / height, this.groundBandTop);
            }
        }

        this.lastCostMs = (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;
        return { columns: out, blockRow, floorMean, costMs: this.lastCostMs };
    }
}

/**
 * Load governor for the refinement stage.
 *
 * Policy:
 *   - Run when an obstacle is present, when the direction decision is close, or
 *     when the scene just changed. Those are the moments extra information can
 *     change an outcome.
 *   - Otherwise run periodically at a low rate, so the grid does not go stale.
 *   - Track the cost. If the stage's own p95 cost exceeds its budget, or the
 *     pipeline's p95 latency regresses while it is on, disable it and say so.
 */
export class AdaptiveSegmentationController {
    /**
     * @param {{estimator:object, enabled?:boolean, budgetMs?:number,
     *          periodicIntervalMs?:number, maxCostShare?:number}} opts
     */
    constructor({
        estimator,
        enabled = true,
        budgetMs = 3.5,
        periodicIntervalMs = 400,
        maxCostShare = 0.18
    }) {
        this.estimator = estimator;
        this.enabled = enabled;
        this.budgetMs = budgetMs;
        this.periodicIntervalMs = periodicIntervalMs;
        this.maxCostShare = maxCostShare;

        this.disabledReason = enabled ? null : "disabled by device profile";
        this._lastRunAt = -Infinity;
        this._costs = [];
        this.runs = 0;
        this.skips = 0;
        this.lastTrigger = SegmentationTrigger.DISABLED;
        this.lastResult = null;
    }

    /** Sorted-copy p95 of recent stage costs. */
    get p95CostMs() {
        if (this._costs.length === 0) return 0;
        const sorted = this._costs.slice().sort((a, b) => a - b);
        return sorted[Math.min(sorted.length - 1, Math.ceil(0.95 * sorted.length) - 1)];
    }

    get meanCostMs() {
        if (this._costs.length === 0) return 0;
        return this._costs.reduce((a, b) => a + b, 0) / this._costs.length;
    }

    disable(reason) {
        if (!this.enabled) return;
        this.enabled = false;
        this.disabledReason = reason;
        this.lastResult = null;
        this.lastTrigger = SegmentationTrigger.DISABLED;
    }

    /**
     * @param {object} ctx
     * @param {Uint8Array} ctx.luma
     * @param {number} ctx.width
     * @param {number} ctx.height
     * @param {number} ctx.nowMs
     * @param {boolean} ctx.obstaclePresent
     * @param {boolean} ctx.directionUncertain
     * @param {boolean} ctx.sceneChanged
     * @param {number} ctx.pipelineP95Ms used to detect that we are the problem
     * @returns {{columns:Float32Array}|null}
     */
    maybeRun(ctx) {
        if (!this.enabled) {
            this.lastTrigger = SegmentationTrigger.DISABLED;
            return null;
        }

        const trigger = this._selectTrigger(ctx);
        if (!trigger) {
            this.skips += 1;
            // Keep serving the previous grid: stale free-space beats none, and
            // the detector-derived grid is fused on top of it every frame anyway.
            return this.lastResult;
        }

        const result = this.estimator.estimate(ctx.luma, ctx.width, ctx.height);
        this.runs += 1;
        this._lastRunAt = ctx.nowMs;
        this.lastTrigger = trigger;

        this._costs.push(result.costMs);
        if (this._costs.length > 40) this._costs.shift();

        // Self-policing: if the stage is either over its own budget or is
        // eating a meaningful share of a pipeline that is already struggling,
        // turn it off rather than let it degrade the walk.
        if (this._costs.length >= 12) {
            const p95 = this.p95CostMs;
            if (p95 > this.budgetMs) {
                this.disable(`stage p95 cost ${p95.toFixed(1)} ms exceeded the ${this.budgetMs} ms budget`);
                return null;
            }
            if (ctx.pipelineP95Ms > 0 && p95 / ctx.pipelineP95Ms > this.maxCostShare) {
                this.disable(
                    `stage cost ${p95.toFixed(1)} ms is ${Math.round(100 * p95 / ctx.pipelineP95Ms)}% of pipeline latency`
                );
                return null;
            }
        }

        this.lastResult = { columns: result.columns, blockRow: result.blockRow };
        return this.lastResult;
    }

    _selectTrigger(ctx) {
        if (ctx.sceneChanged) return SegmentationTrigger.SCENE_CHANGED;
        if (ctx.directionUncertain) return SegmentationTrigger.DIRECTION_UNCERTAIN;
        if (ctx.obstaclePresent) return SegmentationTrigger.OBSTACLE_PRESENT;
        if (ctx.nowMs - this._lastRunAt >= this.periodicIntervalMs) return SegmentationTrigger.PERIODIC;
        return null;
    }

    metrics() {
        return {
            enabled: this.enabled,
            disabledReason: this.disabledReason,
            lastTrigger: this.lastTrigger,
            runs: this.runs,
            skips: this.skips,
            meanCostMs: Number(this.meanCostMs.toFixed(2)),
            p95CostMs: Number(this.p95CostMs.toFixed(2))
        };
    }
}
