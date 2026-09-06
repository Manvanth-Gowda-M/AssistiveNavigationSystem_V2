/**
 * Navigation & Spatial Reasoning Configuration
 * Defines walking corridor boundaries, distance thresholds, danger weights, and hysteresis.
 */

export const NavigationConfig = {
    // Spatial Zones (0.0 to 1.0 normalized horizontal space)
    zones: {
        left: { min: 0.00, max: 0.33, label: "LEFT" },
        center: { min: 0.33, max: 0.67, label: "CENTER" },
        right: { min: 0.67, max: 1.00, label: "RIGHT" }
    },

    // Walking Corridor Geometry
    corridor: {
        leftBoundary: 0.28,         // Left edge of user's walking path
        rightBoundary: 0.72,        // Right edge of user's walking path
        bottomActiveRatio: 0.85,    // Bottom vertical space considered ground/collision trajectory
        minCorridorOverlapRatio: 0.20 // Overlap required to flag an in-corridor obstacle
    },

    // Distance & Proximity Classification (Normalized BBox Height & Area)
    proximityBands: {
        VERY_CLOSE: { minArea: 0.30, minHeight: 0.65, riskMultiplier: 2.5, score: 95 },
        CLOSE:      { minArea: 0.12, minHeight: 0.40, riskMultiplier: 1.8, score: 75 },
        MEDIUM:     { minArea: 0.04, minHeight: 0.20, riskMultiplier: 1.2, score: 45 },
        FAR:        { minArea: 0.00, minHeight: 0.00, riskMultiplier: 0.6, score: 15 }
    },

    // Approach Velocity
    approachRateThreshold: 0.08,    // Area growth rate per second indicating approach
    rapidApproachThreshold: 0.22,   // High rate of area expansion (rapid collision trajectory)

    // Decision Hysteresis
    hysteresis: {
        minClearanceDelta: 0.18,    // Clearance differential required to switch directions
        decisionHoldFrames: 4,      // Number of frames a direction decision is maintained
        directionConfidenceMin: 0.55 // Minimum confidence to give directional advice vs. general caution
    },

    // Free Space Clearance Calculation
    freeSpaceGridSegments: 10       // Quantization slices across horizontal width
};
