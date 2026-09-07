/**
 * Scene-change and camera-turn detection.
 *
 * Why this exists: when you turn a corner, every track the system holds becomes
 * a lie. The obstacle that was "ahead" is now behind your shoulder, but IoU
 * matching will happily keep coasting it into the corridor and the system will
 * warn about something that is no longer in front of you (requirement 67).
 *
 * Two signals, both from the shared analysis thumbnail:
 *   - **change**: mean absolute luminance difference. Detects entering a room,
 *     a vehicle filling the frame, lights coming on.
 *   - **shift**: best horizontal alignment offset between consecutive frames.
 *     A consistent non-zero shift is a pan/turn; a large one is a whip-pan.
 *
 * Column-profile correlation is used for the shift estimate rather than optical
 * flow: it is O(width * maxLag) on a 32-wide profile, which is a few hundred
 * operations, and it is precisely as much fidelity as the decision "should I
 * distrust my tracks?" requires.
 */

import { VisionConfig } from "../config/visionConfig.js";

export const SceneEvent = Object.freeze({
    STABLE: "STABLE",
    /** Content changed substantially without a clear directional pan. */
    SCENE_CHANGE: "SCENE_CHANGE",
    /** Consistent horizontal motion: the user is turning. */
    TURN: "TURN",
    /** Very large motion: whip-pan or the phone being repositioned. */
    VIOLENT_MOTION: "VIOLENT_MOTION"
});

/**
 * Mean absolute difference between two luminance planes, normalised to 0..1.
 * @returns {number}
 */
export function meanAbsoluteDifference(a, b, length) {
    if (!a || !b || length <= 0) return 0;
    let sum = 0;
    for (let i = 0; i < length; i += 1) {
        const d = a[i] - b[i];
        sum += d < 0 ? -d : d;
    }
    return sum / (length * 255);
}

/**
 * Collapse a luminance plane into a per-column mean profile.
 * @param {Uint8Array|number[]} luma
 * @param {Float32Array} out length must equal `width`
 */
export function columnProfile(luma, width, height, out) {
    for (let x = 0; x < width; x += 1) out[x] = 0;
    for (let y = 0; y < height; y += 1) {
        const row = y * width;
        for (let x = 0; x < width; x += 1) out[x] += luma[row + x];
    }
    for (let x = 0; x < width; x += 1) out[x] /= height;
    return out;
}

/**
 * Estimate horizontal shift between two column profiles by minimising SAD over
 * integer lags.
 *
 * @returns {{shiftColumns:number, confidence:number}}
 *          `shiftColumns` is positive when the *content* moved right (i.e. the
 *          camera panned left). `confidence` is how much better the best lag was
 *          than the zero lag, normalised to 0..1.
 */
export function estimateHorizontalShift(prev, curr, width, maxLag) {
    let bestLag = 0;
    let bestCost = Infinity;
    let zeroCost = 0;

    for (let lag = -maxLag; lag <= maxLag; lag += 1) {
        let cost = 0;
        let count = 0;
        const start = Math.max(0, -lag);
        const end = Math.min(width, width - lag);
        for (let x = start; x < end; x += 1) {
            const d = curr[x + lag] - prev[x];
            cost += d < 0 ? -d : d;
            count += 1;
        }
        if (count === 0) continue;
        const normalised = cost / count;
        if (lag === 0) zeroCost = normalised;
        if (normalised < bestCost) {
            bestCost = normalised;
            bestLag = lag;
        }
    }

    // If shifting the frame explains the difference much better than not
    // shifting it, we are looking at a pan rather than content change.
    const improvement = zeroCost > 0 ? (zeroCost - bestCost) / zeroCost : 0;
    return {
        shiftColumns: bestLag,
        confidence: Math.max(0, Math.min(1, improvement))
    };
}

export class SceneChangeDetector {
    constructor(cfg = VisionConfig.sceneChange) {
        this.cfg = cfg;
        this._prevLuma = null;
        this._prevProfile = null;
        this._currProfile = null;
        this._width = 0;
        this._height = 0;

        this.lastChange = 0;
        this.lastShiftFraction = 0;
        this.lastShiftConfidence = 0;
        this.lastEvent = SceneEvent.STABLE;
        /** Accumulated absolute pan since the last stable stretch. */
        this.cumulativeShift = 0;
        this.eventCounts = { [SceneEvent.SCENE_CHANGE]: 0, [SceneEvent.TURN]: 0, [SceneEvent.VIOLENT_MOTION]: 0 };
    }

    _ensureBuffers(width, height) {
        if (this._width === width && this._height === height && this._prevLuma) return;
        this._width = width;
        this._height = height;
        this._prevLuma = new Uint8Array(width * height);
        this._prevProfile = new Float32Array(width);
        this._currProfile = new Float32Array(width);
        this._primed = false;
    }

    /**
     * @param {Uint8Array} luma
     * @returns {{event:string, change:number, shiftFraction:number, shiftConfidence:number,
     *            trustTracks:boolean}}
     */
    update(luma, width, height) {
        this._ensureBuffers(width, height);
        const n = width * height;

        if (!this._primed) {
            this._prevLuma.set(luma.subarray ? luma.subarray(0, n) : luma);
            columnProfile(luma, width, height, this._prevProfile);
            this._primed = true;
            this.lastEvent = SceneEvent.STABLE;
            return {
                event: SceneEvent.STABLE,
                change: 0,
                shiftFraction: 0,
                shiftConfidence: 0,
                trustTracks: true
            };
        }

        const change = meanAbsoluteDifference(this._prevLuma, luma, n);
        columnProfile(luma, width, height, this._currProfile);
        const { shiftColumns, confidence } = estimateHorizontalShift(
            this._prevProfile,
            this._currProfile,
            width,
            Math.min(this.cfg.maxShiftColumns, Math.floor(width / 3))
        );
        const shiftFraction = shiftColumns / width;
        const absShift = Math.abs(shiftFraction);

        let event = SceneEvent.STABLE;
        if (absShift >= this.cfg.rotationShiftThreshold * 2 && confidence > 0.25) {
            event = SceneEvent.VIOLENT_MOTION;
        } else if (absShift >= this.cfg.rotationShiftThreshold && confidence > 0.2) {
            event = SceneEvent.TURN;
        } else if (change >= this.cfg.changeThreshold) {
            event = SceneEvent.SCENE_CHANGE;
        }

        if (event === SceneEvent.STABLE) {
            this.cumulativeShift *= 0.6;
        } else {
            this.cumulativeShift += absShift;
            this.eventCounts[event] += 1;
        }

        // Roll buffers forward. `set` copies into existing storage; no allocation.
        this._prevLuma.set(luma.subarray ? luma.subarray(0, n) : luma);
        const swap = this._prevProfile;
        this._prevProfile = this._currProfile;
        this._currProfile = swap;

        this.lastChange = change;
        this.lastShiftFraction = shiftFraction;
        this.lastShiftConfidence = confidence;
        this.lastEvent = event;

        return {
            event,
            change,
            shiftFraction,
            shiftConfidence: confidence,
            // Tracks survive a plain scene change (content moved but geometry is
            // still roughly valid) but not a turn.
            trustTracks: event === SceneEvent.STABLE || event === SceneEvent.SCENE_CHANGE
        };
    }

    /** True when perception should be treated as high priority for a moment. */
    get needsPerceptionBoost() {
        return this.lastEvent !== SceneEvent.STABLE;
    }

    metrics() {
        return {
            event: this.lastEvent,
            change: Number(this.lastChange.toFixed(3)),
            shift: Number(this.lastShiftFraction.toFixed(3)),
            shiftConfidence: Number(this.lastShiftConfidence.toFixed(2)),
            cumulativeShift: Number(this.cumulativeShift.toFixed(2)),
            counts: { ...this.eventCounts }
        };
    }

    reset() {
        this._primed = false;
        this.cumulativeShift = 0;
        this.lastEvent = SceneEvent.STABLE;
    }
}
