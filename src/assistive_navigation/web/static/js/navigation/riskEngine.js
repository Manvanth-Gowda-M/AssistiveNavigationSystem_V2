/**
 * Risk Engine Module
 * Estimates approximate distance, approach dynamics, and composite collision risk scores (0-100).
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { VisionConfig } from "../config/visionConfig.js";

export class RiskEngine {
    /**
     * Classifies approximate distance based on normalized height & area.
     */
    static estimateDistanceCategory(box, areaNorm) {
        const height = box[3] - box[1];
        const { VERY_CLOSE, CLOSE, MEDIUM } = NavigationConfig.proximityBands;

        if (areaNorm >= VERY_CLOSE.minArea || height >= VERY_CLOSE.minHeight) {
            return "VERY_CLOSE";
        }
        if (areaNorm >= CLOSE.minArea || height >= CLOSE.minHeight) {
            return "CLOSE";
        }
        if (areaNorm >= MEDIUM.minArea || height >= MEDIUM.minHeight) {
            return "MEDIUM";
        }
        return "FAR";
    }

    /**
     * Computes composite collision risk (0 - 100).
     * risk = pathOverlap * proximityFactor * confidenceFactor * approachFactor * priorityFactor
     */
    static computeObstacleRisk(track, corridorEval) {
        const distCat = this.estimateDistanceCategory(track.boundingBox, track.areaNorm);
        track.estimatedDistance = distCat;

        const proximityCfg = NavigationConfig.proximityBands[distCat] || NavigationConfig.proximityBands.FAR;
        let baseScore = proximityCfg.score;

        // Path Overlap Factor (0.2 to 1.0)
        const overlapFactor = corridorEval.inCorridor ? (0.6 + 0.4 * corridorEval.overlapRatio) : 0.25;

        // Semantic Category Priority
        let priorityFactor = 1.0;
        if (VisionConfig.highPriorityLabels.has(track.label)) {
            priorityFactor = 1.35; // Vehicles, stairs, persons
        } else if (VisionConfig.mediumPriorityLabels.has(track.label)) {
            priorityFactor = 1.10;
        }

        // Approach Velocity Factor
        let approachFactor = 1.0;
        if (track.areaGrowthRate > NavigationConfig.rapidApproachThreshold) {
            approachFactor = 1.6; // Approaching very fast!
        } else if (track.isApproaching) {
            approachFactor = 1.25;
        }

        // Confidence Factor
        const confFactor = Math.max(0.5, track.confidence);

        // Calculate Composite Risk Score (0-100)
        let compositeRisk = baseScore * overlapFactor * priorityFactor * approachFactor * confFactor;
        compositeRisk = Math.min(100, Math.max(0, Math.round(compositeRisk)));

        track.estimatedRisk = compositeRisk;
        return compositeRisk;
    }
}
