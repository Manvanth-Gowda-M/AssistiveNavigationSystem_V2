/**
 * Geometric calculations for bounding boxes, corridor intersections, and IoU.
 */

export class GeometryUtils {
    /**
     * Compute Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format.
     */
    static calculateIoU(boxA, boxB) {
        const xA = Math.max(boxA[0], boxB[0]);
        const yA = Math.max(boxA[1], boxB[1]);
        const xB = Math.min(boxA[2], boxB[2]);
        const yB = Math.min(boxA[3], boxB[3]);

        const interWidth = Math.max(0, xB - xA);
        const interHeight = Math.max(0, yB - yA);
        const interArea = interWidth * interHeight;

        const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
        const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
        const unionArea = areaA + areaB - interArea;

        if (unionArea <= 0) return 0;
        return interArea / unionArea;
    }

    /**
     * Calculate Euclidean distance between two 2D normalized centers.
     */
    static centerDistance(c1, c2) {
        const dx = c1.x - c2.x;
        const dy = c1.y - c2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Computes horizontal overlap percentage between a bounding box [x1, y1, x2, y2] and corridor [cLeft, cRight].
     */
    static computeCorridorOverlap(box, cLeft, cRight) {
        const [x1, , x2] = box;
        const boxWidth = x2 - x1;
        if (boxWidth <= 0) return 0;

        const overlapLeft = Math.max(x1, cLeft);
        const overlapRight = Math.min(x2, cRight);
        const overlapWidth = Math.max(0, overlapRight - overlapLeft);

        return overlapWidth / boxWidth;
    }

    /**
     * Calculates free space on left and right sides of the corridor.
     * Returns { leftClearance: 0.0-1.0, rightClearance: 0.0-1.0 }.
     */
    static estimateFlankClearance(objects, corridorLeft, corridorRight) {
        let leftOccupied = 0;
        let rightOccupied = 0;

        for (const obj of objects) {
            const [x1, , x2] = obj.boundingBox;
            const weight = Math.max(0.6, (obj.estimatedRisk || 50) / 100);

            // Left flank: 0.0 to corridorLeft
            if (x1 < corridorLeft) {
                const overlap = Math.max(0, Math.min(x2, corridorLeft) - Math.max(x1, 0.0));
                leftOccupied += (overlap / corridorLeft) * weight;
            }

            // Right flank: corridorRight to 1.0
            if (x2 > corridorRight) {
                const rightFlankWidth = 1.0 - corridorRight;
                const overlap = Math.max(0, Math.min(x2, 1.0) - Math.max(x1, corridorRight));
                rightOccupied += (overlap / rightFlankWidth) * weight;
            }
        }

        return {
            leftClearance: Math.max(0.0, 1.0 - Math.min(1.0, leftOccupied)),
            rightClearance: Math.max(0.0, 1.0 - Math.min(1.0, rightOccupied))
        };
    }
}
