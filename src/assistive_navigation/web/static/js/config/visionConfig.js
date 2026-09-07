/**
 * Vision / perception tuning constants.
 *
 * Everything here is a *starting point* that the runtime is allowed to move
 * (see vision/frameScheduler.js and config/deviceProfiles.js). Nothing in this
 * file should be treated as a hard device capability.
 *
 * Pure data. Importable in Node.
 */

export const VisionConfig = {
    /* ---------------------------------------------------------------- input */

    /**
     * Inference input is a dedicated low-resolution stream, independent of the
     * preview resolution. Overridden per device profile.
     */
    inference: {
        defaultInputSize: 320,
        allowedInputSizes: [256, 320, 384],
        /** Letterbox fill. Matches the value baked into the ONNX export. */
        padValue: 114
    },

    /** Preview stream request. The preview is never the inference source. */
    preview: {
        idealWidth: 960,
        idealHeight: 720,
        minWidth: 320,
        minHeight: 240,
        idealFrameRate: 30
    },

    /* ------------------------------------------------------------ detection */

    /**
     * Raised sharply from the pre-upgrade 0.12-0.15. Low thresholds were the
     * single largest source of direction churn and speech spam: a 0.15
     * confidence blob was allowed to drive a STOP.
     *
     * `scoreThreshold`     - anything below this is discarded at decode time.
     * `acceptThreshold`    - needed to open a new track.
     * `emergencyThreshold` - a detection this confident may drive an emergency
     *                        response without waiting for track confirmation.
     */
    detection: {
        scoreThreshold: 0.30,
        acceptThreshold: 0.38,
        emergencyThreshold: 0.55,
        nmsIouThreshold: 0.45,
        maxDetections: 24,
        /** Boxes smaller than this fraction of frame area are noise. */
        minBoxArea: 0.0015
    },

    /* ------------------------------------------------------------- tracking */

    tracking: {
        /** Matching gates. A pair must pass IoU *or* centroid distance. */
        minMatchIou: 0.25,
        maxMatchCentroidDistance: 0.13,
        /** Frames of consistent association before a track can drive guidance. */
        confirmFrames: 3,
        /** Frames a lost track coasts before deletion. */
        maxCoastFrames: 6,
        /** Hard wall-clock timeout so a stalled pipeline cannot keep ghosts. */
        maxCoastMs: 700,
        /** Exponential smoothing factors. Higher = more responsive. */
        boxAlpha: 0.55,
        confidenceAlpha: 0.45,
        velocityAlpha: 0.40,
        growthAlpha: 0.35,
        /** History depth used for approach-rate regression. */
        historyLength: 10,
        /**
         * After a large scene change (turning a corner) old tracks are unsafe
         * to trust. Their confidence is multiplied by this and confirmation is
         * revoked.
         */
        sceneChangeConfidenceDecay: 0.55
    },

    /* ------------------------------------------------------------ scheduler */

    scheduler: {
        /** Inference cadence bounds, in frames per second. */
        minFps: 5,
        maxFps: 22,
        startFps: 12,
        /**
         * The scheduler steers p95 latency into this window. Below the low
         * bound it speeds up; above the high bound it slows down.
         */
        targetP95LatencyMs: 85,
        maxP95LatencyMs: 140,
        /** Consecutive comfortable windows required before speeding up. */
        rampUpWindows: 3,
        /** How often the controller re-evaluates, in milliseconds. */
        controlIntervalMs: 1000,
        /**
         * Sustained-load guard. If p95 latency degrades by more than this
         * fraction relative to the session's best measured p95, treat it as
         * thermal throttling and shed load rather than collapse.
         */
        thermalRegressionRatio: 1.5,
        /** Latency window size for percentile statistics. */
        latencyWindow: 60
    },

    /* -------------------------------------------------------- frame quality */

    quality: {
        /** Analysis thumbnail. One preallocated canvas, reused forever. */
        sampleWidth: 64,
        sampleHeight: 48,
        /** Rolling window over which the quality score is averaged. */
        window: 8,
        /** Mean luminance (0-255) below/above which the frame is unusable. */
        darkThreshold: 32,
        brightThreshold: 246,
        /**
         * Normalised Laplacian-style gradient energy. Below this the frame is
         * treated as motion-blurred.
         *
         * Calibrated against synthetic 64x48 analysis thumbnails (see
         * tests/js/pipeline.test.mjs): a heavily smeared frame measures around
         * 0.0005, a structured indoor scene around 0.007, and a hard-edged scene
         * around 0.009. 0.0025 sits with clear margin on both sides. Note that a
         * featureless surface with only a lighting gradient also measures ~0,
         * which is correct to flag - detection on a blank wall is not reliable
         * either.
         */
        blurThreshold: 0.0025,
        /** Luminance std-dev below this means a featureless frame. */
        lowContrastThreshold: 8,
        /**
         * Consecutive poor frames tolerated before the system downgrades. One
         * or two bad frames while walking is normal and must not trigger an
         * alarm (requirements 39-41).
         */
        degradeAfterPoorFrames: 4,
        recoverAfterGoodFrames: 3,
        /** Quality score (0-1) under which directional advice is withheld. */
        minScoreForDirection: 0.45,
        /** Quality score under which the system asks the user to stop. */
        minScoreForOperation: 0.22
    },

    /* --------------------------------------------------------- scene change */

    sceneChange: {
        sampleWidth: 32,
        sampleHeight: 24,
        /** Mean absolute luminance difference (0-1) indicating a new scene. */
        changeThreshold: 0.16,
        /** Estimated horizontal shift (fraction of width) indicating a turn. */
        rotationShiftThreshold: 0.14,
        /** Max lag searched when estimating horizontal shift, in columns. */
        maxShiftColumns: 8
    },

    /* --------------------------------------------------------- health/watch */

    health: {
        /** No successful inference for this long triggers recovery. */
        watchdogTimeoutMs: 2500,
        /** Consecutive inference failures that trigger recovery. */
        maxConsecutiveFailures: 4,
        /** Recovery attempts before declaring the vision system unavailable. */
        maxRecoveryAttempts: 4,
        /** Backoff between recovery attempts. */
        recoveryBackoffMs: [250, 750, 1500, 3000],
        /** Heap growth (bytes) across a monitoring window that we flag. */
        heapGrowthWarningBytes: 96 * 1024 * 1024,
        heapSampleIntervalMs: 5000,
        heapSampleWindow: 24
    },

    /* ------------------------------------------------------------ warm-up */

    warmup: {
        /** Inference passes run before the user is told the system is ready. */
        passes: 4,
        /** A warm-up pass slower than this fails the health check. */
        maxAcceptableLatencyMs: 700,
        /** Overall warm-up budget. */
        timeoutMs: 20000
    }
};

export default VisionConfig;
