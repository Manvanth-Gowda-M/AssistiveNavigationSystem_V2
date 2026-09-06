/**
 * Walking Corridor Module
 * Analyzes the user's projected walking path and evaluates obstacle corridor overlap.
 */

import { NavigationConfig } from "../config/navigationConfig.js";
import { GeometryUtils } from "../utils/geometry.js";

export class WalkingCorridor {
    constructor(config = NavigationConfig.corridor) {
        this.leftBoundary = config.leftBoundary;
        this.rightBoundary = config.rightBoundary;
        this.bottomActiveRatio = config.bottomActiveRatio;
        this.minOverlapRatio = config.minCorridorOverlapRatio;
    }

    /**
     * Evaluates whether an object intersects the walking path.
     * Returns { inCorridor: boolean, overlapRatio: number, isCenterBlocked: boolean }
     */
    evaluateObstacle(box) {
        const [x1, y1, x2, y2] = box;
        const overlapRatio = GeometryUtils.computeCorridorOverlap(box, this.leftBoundary, this.rightBoundary);
        
        // Vertical check: only obstacles occupying the ground/middle area are immediate path obstructions
        const isGroundRelevance = y2 >= (1.0 - this.bottomActiveRatio);

        const inCorridor = (overlapRatio >= this.minOverlapRatio) && isGroundRelevance;
        const centerX = (x1 + x2) / 2;
        const isCenterBlocked = inCorridor && (centerX >= this.leftBoundary && centerX <= this.rightBoundary);

        return {
            inCorridor,
            overlapRatio,
            isCenterBlocked
        };
    }

    getBounds() {
        return {
            left: this.leftBoundary,
            right: this.rightBoundary,
            width: this.rightBoundary - this.leftBoundary
        };
    }
}
