/**
 * Speech configuration.
 *
 * Two rules govern everything here:
 *   - **Short.** Every phrase is an instruction, not a description. "Move
 *     slightly left." not "Chair ahead on your left, move slightly left."
 *     The user is walking; they need two words, not a sentence.
 *   - **Silence is a feature.** The default output for a clear path is nothing
 *     at all (requirement 37).
 *
 * Pure data. Importable in Node.
 */

export const SpeechPriority = Object.freeze({
    STOP: 1,
    IMMEDIATE_DANGER: 2,
    DIRECTION: 3,
    CAUTION: 4,
    OBJECT: 5,
    PATH_STATUS: 6,
    SYSTEM: 7
});

export const SpeechConfig = {
    Priority: SpeechPriority,

    cooldowns: {
        /**
         * Emergency speech ignores every other cooldown; this is the floor that
         * stops a stuttering "Stop. Stop. Stop." while the hazard persists.
         *
         * 2 s rather than 1.2 s: the user needs one clear instruction and then
         * time to act on it. Re-asserting a stop three times a second is not more
         * safety, it is noise that buries the next real instruction.
         */
        emergencyMs: 2000,
        /** Repeating the *same* instruction. Long, because repetition is noise. */
        repeatSameMs: 3500,
        /** Switching to a *different* instruction. */
        changeActionMs: 1100,
        /** Caution-level messages. */
        cautionMs: 3000,
        /** "Path clear." Announced once per obstruction episode anyway. */
        pathStatusMs: 6000,
        /** Visibility warnings. */
        visibilityMs: 9000,
        /** System/status messages. */
        systemMs: 800
    },

    /**
     * A held instruction is re-spoken this often *only* while the situation is
     * still live, so a user who missed it gets a second chance without being
     * nagged.
     */
    reminderMs: 5000,

    /** Risk change (0-100 points) that counts as a material change. */
    materialRiskDelta: 18,

    phrases: {
        STOP: "Stop.",
        STOP_BLOCKED: "Stop. Path blocked.",
        STOP_VERY_NEAR: "Stop. Obstacle close.",
        STOP_APPROACHING: "Stop. Object approaching.",
        STOP_VEHICLE: "Stop. Vehicle.",
        STOP_UNRELIABLE: "Please stop. Vision unreliable.",

        MOVE_LEFT: "Move left.",
        MOVE_SLIGHTLY_LEFT: "Move slightly left.",
        MOVE_RIGHT: "Move right.",
        MOVE_SLIGHTLY_RIGHT: "Move slightly right.",

        OBSTACLE_AHEAD: "Obstacle ahead.",
        PERSON_AHEAD: "Person ahead.",
        PERSON_CROSSING: "Person crossing.",
        VEHICLE_APPROACHING: "Vehicle approaching.",
        CAUTION: "Obstacle ahead. Use caution.",
        NARROW: "Narrow path.",

        PATH_CLEAR: "Path clear.",

        LOW_VISIBILITY: "Visibility is poor.",
        CAMERA_BLOCKED: "Camera blocked.",

        SYSTEM_READY: "Vision assistance ready.",
        ASSISTANCE_STARTED: "Assistance started.",
        ASSISTANCE_PAUSED: "Paused.",
        ASSISTANCE_RESUMED: "Resumed.",
        ASSISTANCE_STOPPED: "Assistance stopped.",
        CAMERA_UNAVAILABLE: "Camera unavailable.",
        VISION_UNAVAILABLE: "Vision system unavailable. Please stop.",
        RECOVERING: "Reconnecting vision.",
        DEGRADED: "Reduced accuracy."
    },

    /** Speech synthesis defaults. Brisk but not rushed. */
    voice: {
        rate: 1.12,
        pitch: 1.0,
        volume: 1.0,
        preferredLanguage: "en-US"
    },

    /**
     * Maximum utterances per minute, as a hard backstop independent of all the
     * per-message logic. If this ever engages something upstream is wrong, and
     * it is logged rather than silently applied.
     */
    maxUtterancesPerMinute: 18
};

export default SpeechConfig;
