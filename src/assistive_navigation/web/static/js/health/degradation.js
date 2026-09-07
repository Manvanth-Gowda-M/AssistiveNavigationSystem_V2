/**
 * Graceful degradation ladder (requirement 49).
 *
 * When something stops working, the system sheds capability instead of failing.
 * Each level is a coherent, safe operating mode - not a broken version of the
 * level above it.
 *
 *   LEVEL 1  detector + tracking + free-space refinement
 *            Full pipeline. Directional guidance with free-space reasoning.
 *
 *   LEVEL 2  detector + tracking
 *            Refinement stage disabled (too slow, or it failed). Free space is
 *            still estimated from detections, so directions remain available.
 *
 *   LEVEL 3  detector only
 *            Tracking is unreliable (e.g. the frame rate is too low for
 *            association to work). No velocity, no approach detection, so no
 *            confident directions - obstacle warnings and STOP only.
 *
 *   LEVEL 4  unreliable
 *            No trustworthy perception. Say so and tell the user to stop.
 *
 * The important property: **level 3 changes what the system is willing to say**,
 * not just what it computes. A system that keeps issuing confident "move left"
 * instructions from untrustworthy data is more dangerous than one that admits it
 * cannot see.
 */

/**
 * Completed inferences required before the inference-rate check is trusted.
 * Below this the rolling FPS window is still filling and reads artificially low.
 */
const MIN_SAMPLES_FOR_RATE = 12;

export const DegradationLevel = Object.freeze({
    FULL: 1,
    NO_REFINEMENT: 2,
    DETECTOR_ONLY: 3,
    UNRELIABLE: 4
});

const LEVEL_CAPABILITIES = Object.freeze({
    [DegradationLevel.FULL]: {
        label: "Full pipeline",
        detector: true,
        tracking: true,
        freeSpaceRefinement: true,
        directionalGuidance: true,
        approachDetection: true,
        description: "Detector, tracking and free-space refinement all active."
    },
    [DegradationLevel.NO_REFINEMENT]: {
        label: "Detector + tracking",
        detector: true,
        tracking: true,
        freeSpaceRefinement: false,
        directionalGuidance: true,
        approachDetection: true,
        description: "Free-space refinement disabled to protect frame rate."
    },
    [DegradationLevel.DETECTOR_ONLY]: {
        label: "Detector only",
        detector: true,
        tracking: false,
        freeSpaceRefinement: false,
        directionalGuidance: false,
        approachDetection: false,
        description: "Tracking unreliable. Warnings and stops only, no directions."
    },
    [DegradationLevel.UNRELIABLE]: {
        label: "Unreliable",
        detector: false,
        tracking: false,
        freeSpaceRefinement: false,
        directionalGuidance: false,
        approachDetection: false,
        description: "Vision system unavailable."
    }
});

export class DegradationController {
    /**
     * @param {{startLevel?:number, onChange?:(level:number, detail:object)=>void}} [opts]
     */
    constructor({ startLevel = DegradationLevel.FULL, onChange = () => {} } = {}) {
        this.level = startLevel;
        this.onChange = onChange;
        this.reasons = [];
        this.history = [];
    }

    get capabilities() {
        return LEVEL_CAPABILITIES[this.level];
    }

    get label() {
        return LEVEL_CAPABILITIES[this.level].label;
    }

    /** Drop to at least `level`. Never silently improves. */
    degradeTo(level, reason) {
        if (level <= this.level) return false;
        const from = this.level;
        this.level = level;
        this.reasons.push({ level, reason, at: Date.now() });
        if (this.reasons.length > 20) this.reasons.shift();
        this.history.push({ from, to: level, reason, at: Date.now() });
        this.onChange(level, { from, reason, capabilities: this.capabilities });
        return true;
    }

    /**
     * Restore capability. Requires an explicit reason, because silently
     * upgrading after a failure is how you get a system that oscillates between
     * modes.
     */
    restoreTo(level, reason) {
        if (level >= this.level) return false;
        const from = this.level;
        this.level = level;
        this.history.push({ from, to: level, reason, at: Date.now() });
        this.onChange(level, { from, reason, capabilities: this.capabilities });
        return true;
    }

    /**
     * Derive the appropriate level from live conditions.
     *
     * @param {object} conditions
     * @param {boolean} conditions.detectorWorking
     * @param {boolean} conditions.refinementEnabled
     * @param {number} conditions.inferenceFps
     * @param {number} conditions.trackingStability mean track stability 0..1
     * @param {boolean} conditions.perceptionUsable frame quality verdict
     * @param {number} [conditions.completedInferences] guards against judging a
     *        rate before enough samples exist to measure one
     * @returns {{level:number, reason:string}}
     */
    static evaluate(conditions) {
        const {
            detectorWorking,
            refinementEnabled,
            inferenceFps,
            trackingStability,
            perceptionUsable,
            completedInferences = Infinity
        } = conditions;

        // Level 4 is reserved for the system being broken. A detector that is not
        // producing results qualifies; a dark room does not.
        if (!detectorWorking) {
            return { level: DegradationLevel.UNRELIABLE, reason: "detector not producing results" };
        }

        /**
         * Poor frame quality is an *environmental* condition, not a system
         * failure, and conflating the two was a real defect: pointing the camera
         * at a blank wall drove the app to level 4, which enters the error state
         * and demands a manual retry. Walking past a dark doorway must not require
         * the user to restart the vision system.
         *
         * So unusable frames cap at level 3: the detector keeps running, no
         * directions are offered, the user is told visibility is poor, and the
         * system recovers by itself the moment the view improves.
         */
        if (!perceptionUsable) {
            return { level: DegradationLevel.DETECTOR_ONLY, reason: "frame quality unusable" };
        }

        /**
         * The rate check needs enough samples to be a rate.
         *
         * `inferenceFps` is measured over a short rolling window, so for the first
         * second of a session it legitimately reads 1-3 FPS while the window fills.
         * Judging it immediately downgraded to level 3 and back on almost every
         * start, and each downgrade spoke "Reduced accuracy." - three times in a
         * ten-second run. Waiting for a real measurement removes the flapping
         * without weakening the check.
         */
        const rateIsMeasurable = completedInferences >= MIN_SAMPLES_FOR_RATE;

        // Below roughly 4 inference FPS, consecutive frames are 250 ms apart. A
        // person walking at 1.4 m/s moves 35 cm between frames, which breaks
        // both IoU association and any velocity estimate. Tracking output at
        // that rate is not evidence, so we stop acting on it.
        if (rateIsMeasurable && inferenceFps > 0 && inferenceFps < 4) {
            return {
                level: DegradationLevel.DETECTOR_ONLY,
                reason: `inference rate ${inferenceFps.toFixed(1)} FPS too low for tracking`
            };
        }
        if (trackingStability !== null && trackingStability < 0.35) {
            return { level: DegradationLevel.DETECTOR_ONLY, reason: "track association unstable" };
        }
        if (!refinementEnabled) {
            return { level: DegradationLevel.NO_REFINEMENT, reason: "free-space refinement disabled" };
        }
        return { level: DegradationLevel.FULL, reason: "all stages healthy" };
    }

    /**
     * Apply an evaluation. Degrades immediately; upgrades only when the
     * evaluation has agreed for several consecutive calls, so a single good
     * second cannot undo a considered downgrade.
     */
    apply(evaluation, { upgradeConfirmations = 5 } = {}) {
        if (evaluation.level > this.level) {
            this._upgradeStreak = 0;
            return this.degradeTo(evaluation.level, evaluation.reason);
        }
        if (evaluation.level < this.level) {
            this._upgradeStreak = (this._upgradeStreak || 0) + 1;
            if (this._upgradeStreak >= upgradeConfirmations) {
                this._upgradeStreak = 0;
                return this.restoreTo(evaluation.level, evaluation.reason);
            }
            return false;
        }
        this._upgradeStreak = 0;
        return false;
    }

    metrics() {
        return {
            level: this.level,
            label: this.label,
            capabilities: this.capabilities,
            recentChanges: this.history.slice(-4)
        };
    }
}

export { LEVEL_CAPABILITIES };
