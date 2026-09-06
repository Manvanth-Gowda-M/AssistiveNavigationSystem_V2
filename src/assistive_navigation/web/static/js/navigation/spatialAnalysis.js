/**
 * Spatial Analysis Module
 * Classifies objects into Left / Center / Right zones and computes flank clearance.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { GeometryUtils } from "../utils/geometry.js";

export class SpatialAnalysis {
    /**
     * Determine primary direction zone for an object.
     */
    static getDirectionZone(centerX) {
        const { left, center, right } = NavigationConfig.zones;
        if (centerX < left.max) return left.label;
        if (centerX > right.min) return right.label;
        return center.label;
    }

    /**
     * Evaluates open free-space clearance across Left and Right flanks.
     */
    static evaluateFlankClearance(trackedObjects, corridorBounds) {
        return GeometryUtils.estimateFlankClearance(
            trackedObjects,
            corridorBounds.left,
            corridorBounds.right
        );
    }
}
