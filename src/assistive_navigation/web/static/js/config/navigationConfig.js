/**
 * Navigation, spatial reasoning and decision constants.
 *
 * Pure data. Importable in Node.
 *
 * A note on the numbers: none of them are physical measurements. They are
 * image-space heuristics calibrated for a chest/hand-held rear camera at roughly
 * 60-70 degrees horizontal field of view. That is why nothing in this system
 * claims a distance in metres.
 */

export const NavigationConfig = {
    /* ------------------------------------------------------------- corridor */

    corridor: {
        /**
         * Starting walking-corridor half-width, as a fraction of frame width
         * either side of centre. The corridor adapts at runtime
         * (navigation/corridorModel.js); this is the conservative seed.
         */
        baseHalfWidth: 0.20,
        minHalfWidth: 0.15,
        /**
         * 0.24 gives a 48%-of-frame corridor at its widest. Wider than this and
         * objects well off to the side start counting as "in the path", which
         * makes the system warn about things the user will walk past.
         */
        maxHalfWidth: 0.24,
        /**
         * Vertical extent treated as "the ground in front of me". Above
         * `farBoundary` the image is distance/sky and only very large objects
         * matter; below `nearBoundary` is within a stride.
         */
        farBoundary: 0.42,
        nearBoundary: 0.78,
        /** Corridor overlap fraction below which an object is not in the path. */
        minOverlapRatio: 0.12,
        /**
         * How fast the corridor is allowed to move per update. Slow, because a
         * corridor that jumps around produces exactly the direction churn we are
         * eliminating.
         */
        adaptRate: 0.12,
        /**
         * Hard clamp on how far the corridor centre may ever sit from the middle
         * of the frame. Currently the centre does not move at all - see the note
         * in corridorModel.js about why chasing free space caused oscillation -
         * but the clamp stays so that a future orientation-driven centre cannot
         * point the user sideways.
         */
        maxCentreOffset: 0.06
    },

    /* ----------------------------------------------------------- free space */

    freeSpace: {
        /** Horizontal quantisation of the occupancy grid. */
        columns: 24,
        /**
         * Contiguous free width, as a fraction of the frame, that a person needs
         * to walk through. Roughly a shoulder width at conversational distance.
         */
        minPassableWidth: 0.17,
        /** Column free-depth above which the column counts as walkable. */
        passableDepthThreshold: 0.42,
        /**
         * Weighting between average free depth and worst-case free depth when
         * scoring a sector. Worst case is weighted heavily on purpose: an
         * average-clear sector with one blocked column is not clear.
         */
        meanWeight: 0.45,
        minWeight: 0.55,
        /** Ground band within which an obstacle blocks the floor. */
        groundBandTop: 0.42
    },

    /* -------------------------------------------------------- distance bands */

    /**
     * Approximate distance bands. Deliberately coarse and never presented as
     * metres (requirement 25). `bottomY` is the ground-contact cue and is the
     * primary signal; `area` is the secondary signal.
     */
    proximity: {
        VERY_NEAR: { minBottomY: 0.86, minArea: 0.26, proximity: 1.00, label: "very near" },
        NEAR: { minBottomY: 0.74, minArea: 0.13, proximity: 0.78, label: "near" },
        MEDIUM: { minBottomY: 0.58, minArea: 0.045, proximity: 0.50, label: "medium" },
        FAR: { minBottomY: 0.00, minArea: 0.000, proximity: 0.22, label: "far" }
    },

    /* --------------------------------------------------------------- risk */

    risk: {
        /** Approach-rate multipliers. */
        approachFactor: 1.30,
        rapidApproachFactor: 1.70,
        recedingFactor: 0.72,
        /** Lateral motion into the corridor. */
        convergingFactor: 1.22,
        divergingFactor: 0.80,
        /** Minimum weight given to a barely-stable track. */
        minStabilityFactor: 0.55,
        /** Risk at or above this is treated as a critical, immediate threat. */
        criticalThreshold: 78,
        /** Risk at or above this warrants an avoidance manoeuvre. */
        avoidThreshold: 46,
        /** Risk below this is not worth mentioning. */
        noticeThreshold: 26
    },

    /* -------------------------------------------------------- path scoring */

    pathScore: {
        /**
         * Weights for combining free space and obstacle risk into a 0-100 path
         * score. Free space dominates deliberately (requirement 65).
         */
        freeSpaceWeight: 0.60,
        riskWeight: 0.40,
        /** A sector narrower than a shoulder width is unusable regardless of score. */
        impassablePenalty: 0.25,
        /** Score at or above this counts as a safe path. */
        safeThreshold: 55,
        /** Score below this counts as blocked. */
        blockedThreshold: 32,
        /** Deviating from straight ahead has a cost; do not steer for nothing. */
        lateralMoveCost: 8,
        /** Large deviation ("move left") costs more than a small one. */
        largeMoveCost: 14
    },

    /* --------------------------------------------------------- hysteresis */

    hysteresis: {
        /**
         * A new direction must beat the held direction's path score by this
         * margin before the system will switch. This single number is what
         * stops LEFT/RIGHT/LEFT oscillation (requirement 30).
         */
        switchMarginPoints: 14,
        /** Minimum time a non-critical direction is held, in milliseconds. */
        holdMs: 1400,
        /** Minimum time before returning to CONTINUE after a manoeuvre. */
        clearHoldMs: 900,
        /** Consecutive frames of agreement required to adopt a new direction. */
        confirmFrames: 2,
        /** Decision confidence below which we advise caution instead of a side. */
        minDirectionConfidence: 0.52
    },

    /* -------------------------------------------------------- path status */

    pathClear: {
        /**
         * "Path clear" is only spoken after an obstacle state, and only once the
         * path has stayed clear this long (requirement 38).
         */
        stabilityMs: 1600
    }
};

export default NavigationConfig;
