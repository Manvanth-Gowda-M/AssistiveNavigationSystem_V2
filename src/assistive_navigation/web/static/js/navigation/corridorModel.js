/**
 * Dynamic walking corridor.
 *
 * The corridor is the region of the image that corresponds to "where I am about
 * to put my feet". The pre-upgrade build used a fixed 0.28-0.72 rectangle, which
 * is wrong in two ways: it does not follow where the walkable floor actually is,
 * and it does not narrow when the environment gets tight.
 *
 * What adapts, and from what evidence:
 *
 *   centre      <- the centre of the widest passable gap, moved slowly and
 *                  clamped near the middle. The corridor may lean towards an
 *                  opening but must never point off to one side, or the system
 *                  would start "aiming" the user sideways.
 *   half-width  <- how cluttered the near field is. Tight spaces get a narrow
 *                  corridor so the user is not told to squeeze past things.
 *   far / near  <- the observed extent of walkable floor, which is a proxy for
 *                  camera pitch. Pointing the phone down puts the horizon high
 *                  in the frame and shortens the useful corridor.
 *
 * Everything moves at `adaptRate` per update. A corridor that snaps around
 * produces the same direction churn we are engineering out of the system.
 *
 * Pure. Unit-testable.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { Sector } from "../perception/freeSpace.js";

function clamp(v, lo, hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

function approach(current, target, rate) {
    return current + (target - current) * rate;
}

export class CorridorModel {
    constructor(cfg = NavigationConfig.corridor) {
        this.cfg = cfg;
        this.centre = 0.5;
        this.halfWidth = cfg.baseHalfWidth;
        this.farBoundary = cfg.farBoundary;
        this.nearBoundary = cfg.nearBoundary;
        this.clutter = 0;
        this.adaptations = 0;
    }

    get left() {
        return clamp(this.centre - this.halfWidth, 0, 1);
    }

    get right() {
        return clamp(this.centre + this.halfWidth, 0, 1);
    }

    get bounds() {
        return { left: this.left, right: this.right, centre: this.centre, width: this.halfWidth * 2 };
    }

    /**
     * @param {object} ctx
     * @param {object|null} ctx.freeSpace latest free-space report
     * @param {Array} ctx.tracks
     * @param {number} [ctx.aspect] visible-region aspect ratio
     * @param {boolean} [ctx.reset] snap back to defaults (after a turn)
     */
    update({ freeSpace = null, tracks = [], aspect = 4 / 3, reset = false } = {}) {
        const cfg = this.cfg;

        if (reset) {
            this.centre = approach(this.centre, 0.5, 0.5);
            this.halfWidth = approach(this.halfWidth, cfg.baseHalfWidth, 0.5);
            this.farBoundary = approach(this.farBoundary, cfg.farBoundary, 0.5);
            this.nearBoundary = approach(this.nearBoundary, cfg.nearBoundary, 0.5);
            return this.bounds;
        }

        const rate = cfg.adaptRate;

        /* ------------------------------------------------------------ centre */

        /**
         * The corridor centre stays straight ahead.
         *
         * An earlier version steered the centre towards the widest free-space
         * gap. That is wrong, and measurably so: the corridor defines the LEFT,
         * CENTRE and RIGHT sector boundaries, so moving it towards an opening
         * shrinks the sector on that side, lowers that sector's passable width,
         * and flips the path scores. The result was a feedback loop that produced
         * exactly the LEFT/RIGHT/LEFT oscillation this system exists to avoid.
         *
         * Choosing which way to go is the decision engine's job. The corridor's
         * job is only to say where the user is currently heading, which - absent
         * a reliable yaw signal - is straight ahead. `maxCentreOffset` and this
         * field are retained so a real orientation input can drive the centre
         * later without reintroducing the loop.
         */
        const targetCentre = clamp(0.5, 0.5 - cfg.maxCentreOffset, 0.5 + cfg.maxCentreOffset);
        this.centre = approach(this.centre, targetCentre, rate);

        /* --------------------------------------------------------- half-width */

        // Clutter: how much of the near field is occupied. Uses the free-space
        // grid when available because it accounts for unclassified obstacles too.
        let clutter = 0;
        if (freeSpace?.freeDepth) {
            const depth = freeSpace.freeDepth;
            let blocked = 0;
            for (let i = 0; i < depth.length; i += 1) {
                if (depth[i] < NavigationConfig.freeSpace.passableDepthThreshold) blocked += 1;
            }
            clutter = blocked / depth.length;
        } else {
            let nearCount = 0;
            for (const t of tracks) {
                if (t.bottomY >= this.farBoundary) nearCount += 1;
            }
            clutter = clamp(nearCount / 6, 0, 1);
        }
        this.clutter = approach(this.clutter, clutter, rate * 2);

        // Wide when open, narrow when tight.
        const targetHalfWidth = cfg.maxHalfWidth - (cfg.maxHalfWidth - cfg.minHalfWidth) * this.clutter;
        this.halfWidth = approach(this.halfWidth, clamp(targetHalfWidth, cfg.minHalfWidth, cfg.maxHalfWidth), rate);

        /* ----------------------------------------------------- vertical extent */

        // How high up the frame the floor remains open, averaged over the
        // corridor columns. High open floor => the camera is looking further
        // ahead => the corridor can extend further up the image.
        let openness = 0.5;
        if (freeSpace?.freeDepth) {
            const depth = freeSpace.freeDepth;
            const n = depth.length;
            const from = Math.max(0, Math.floor(this.left * n));
            const to = Math.min(n, Math.ceil(this.right * n));
            let sum = 0;
            let count = 0;
            for (let i = from; i < to; i += 1) {
                sum += depth[i];
                count += 1;
            }
            openness = count > 0 ? sum / count : 0.5;
        }

        // A wide (landscape) visible region sees more ground per row, so the
        // useful band starts a little higher.
        const aspectAdjust = clamp((aspect - 1.0) * 0.03, -0.03, 0.05);
        const targetFar = clamp(cfg.farBoundary - (openness - 0.5) * 0.12 - aspectAdjust, 0.30, 0.58);
        this.farBoundary = approach(this.farBoundary, targetFar, rate);
        this.nearBoundary = approach(this.nearBoundary, clamp(cfg.nearBoundary, this.farBoundary + 0.15, 0.92), rate);

        this.adaptations += 1;
        return this.bounds;
    }

    /**
     * Evaluate one box against the corridor.
     *
     * `groundRelevance` is the key output and the thing the old implementation
     * lacked: a box high in the frame is far away or overhead, and must not
     * carry the same weight as one at your feet even if it overlaps the corridor
     * horizontally (requirement 24).
     *
     * @param {number[]} box [x1,y1,x2,y2] navigation coordinates
     * @returns {{inCorridor:boolean, overlapRatio:number, corridorCoverage:number,
     *            groundRelevance:number, sector:string, centreBlocked:boolean}}
     */
    evaluate(box) {
        const [x1, y1, x2, y2] = box;
        const left = this.left;
        const right = this.right;

        const boxWidth = Math.max(1e-6, x2 - x1);
        const overlapWidth = Math.max(0, Math.min(x2, right) - Math.max(x1, left));

        /** Fraction of the *object* inside the corridor. */
        const overlapRatio = overlapWidth / boxWidth;
        /** Fraction of the *corridor* the object blocks. Different question. */
        const corridorCoverage = overlapWidth / Math.max(1e-6, right - left);

        // Ground relevance ramps from 0 at the far boundary to 1 at the near
        // boundary, using the box's bottom edge.
        const span = Math.max(1e-6, this.nearBoundary - this.farBoundary);
        let groundRelevance = (y2 - this.farBoundary) / span;
        groundRelevance = clamp(groundRelevance, 0, 1);

        // A very large object above the far boundary is still relevant: a bus or
        // a wall does not become harmless because its bottom edge is high.
        const area = boxWidth * Math.max(0, y2 - y1);
        if (groundRelevance < 0.35 && area > 0.24) {
            groundRelevance = Math.max(groundRelevance, 0.45);
        }

        const centreX = (x1 + x2) / 2;
        const sector = centreX < left ? Sector.LEFT : (centreX > right ? Sector.RIGHT : Sector.CENTER);

        const inCorridor = overlapRatio >= this.cfg.minOverlapRatio && groundRelevance > 0.05;

        return {
            inCorridor,
            overlapRatio,
            corridorCoverage,
            groundRelevance,
            sector,
            centreBlocked: inCorridor && sector === Sector.CENTER
        };
    }

    metrics() {
        return {
            left: Number(this.left.toFixed(3)),
            right: Number(this.right.toFixed(3)),
            centre: Number(this.centre.toFixed(3)),
            halfWidth: Number(this.halfWidth.toFixed(3)),
            farBoundary: Number(this.farBoundary.toFixed(3)),
            nearBoundary: Number(this.nearBoundary.toFixed(3)),
            clutter: Number(this.clutter.toFixed(2))
        };
    }
}
