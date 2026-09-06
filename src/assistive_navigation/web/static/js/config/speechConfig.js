/**
 * Speech Engine & Natural Language Guidance Configuration
 * Contains prioritized phrase libraries, cooldowns, and speech synthesis parameters.
 */

export const SpeechConfig = {
    // Action Priorities (Lower number = Higher priority)
    Priority: {
        STOP: 1,
        IMMEDIATE_COLLISION: 2,
        HIGH_RISK_OBSTACLE: 3,
        DIRECTION_CHANGE: 4,
        CAUTION: 5,
        OBJECT_DESCRIPTION: 6,
        PATH_STATUS: 7
    },

    // Speech Cooldowns (milliseconds)
    cooldowns: {
        emergencyStopMs: 600,       // Emergency STOP bypasses standard cooldown
        standardInstructionMs: 2400, // Wait time before repeating same direction
        directionChangeMinMs: 900,  // Minimum gap between directional switches
        pathClearMs: 5000,          // Periodic reassurance when path remains clear
        lowVisibilityMs: 8000       // Warning for poor lighting or camera obstruction
    },

    // Deterministic Command Phrase Library (Short, Calm, Actionable)
    phrases: {
        STOP: "Stop.",
        STOP_VERY_CLOSE: "Stop. Obstacle very close.",
        STOP_RAPID_APPROACH: "Stop. Obstacle approaching rapidly.",
        STOP_BOTH_BLOCKED: "Stop. Path blocked.",
        
        MOVE_LEFT: "Move left.",
        MOVE_SLIGHTLY_LEFT: "Move slightly left.",
        MOVE_RIGHT: "Move right.",
        MOVE_SLIGHTLY_RIGHT: "Move slightly right.",
        
        OBSTACLE_AHEAD: "Obstacle ahead.",
        PERSON_AHEAD: "Person ahead.",
        PERSON_CROSSING: "Person crossing ahead.",
        VEHICLE_APPROACHING: "Stop. Vehicle approaching.",
        
        CAUTION_AHEAD: "Obstacle ahead. Use caution.",
        PATH_CLEAR: "Path clear.",
        
        LOW_VISIBILITY: "Low visibility.",
        CAMERA_BLOCKED: "Camera view blocked.",
        
        SYSTEM_READY: "Vision assistance ready.",
        ASSISTANCE_STARTED: "Assistance started.",
        ASSISTANCE_PAUSED: "Assistance paused.",
        ASSISTANCE_STOPPED: "Assistance stopped.",
        CAMERA_UNAVAILABLE: "Camera unavailable.",
        VISION_UNAVAILABLE: "Vision system unavailable."
    },

    // Synthesis Voice Defaults
    defaults: {
        rate: 1.05,                 // Slightly brisk, clear delivery
        pitch: 1.0,                 // Neutral calm pitch
        volume: 1.0,
        preferredLanguage: "en-US"
    }
};
