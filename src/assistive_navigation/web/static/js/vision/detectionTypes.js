/**
 * Normalized Detection Data Structure
 * Standardizes object perception across any detector backend (MediaPipe, TFLite, Server).
 */

export class NormalizedDetection {
    constructor({
        id = null,
        label = "unknown",
        confidence = 0.0,
        boundingBox = [0, 0, 0, 0], // [x1, y1, x2, y2] normalized 0.0 - 1.0
        timestamp = Date.now()
    }) {
        this.id = id;
        this.label = label.toLowerCase();
        this.confidence = Math.max(0.0, Math.min(1.0, confidence));
        this.boundingBox = boundingBox; // [x1, y1, x2, y2]
        
        // Calculated Normalized Geometries
        this.x1 = boundingBox[0];
        this.y1 = boundingBox[1];
        this.x2 = boundingBox[2];
        this.y2 = boundingBox[3];
        this.width = Math.max(0, this.x2 - this.x1);
        this.height = Math.max(0, this.y2 - this.y1);
        this.centerX = this.x1 + this.width / 2;
        this.centerY = this.y1 + this.height / 2;
        this.areaNorm = this.width * this.height;

        // Relative Spatial Properties
        this.relativeHorizontalPosition = this.computeHorizontalZone(this.centerX);
        this.relativeVerticalPosition = this.centerY < 0.33 ? "TOP" : (this.centerY < 0.67 ? "MIDDLE" : "BOTTOM");

        // Dynamic State (Assigned during tracking & risk estimation)
        this.estimatedDistance = "UNKNOWN"; // VERY_CLOSE | CLOSE | MEDIUM | FAR | UNKNOWN
        this.estimatedRisk = 0;             // 0 - 100
        this.trackingConfidence = confidence;
        this.approachVelocity = 0.0;        // Area expansion rate
        this.isApproaching = false;
        this.framesConfirmed = 1;
        this.timestamp = timestamp;
    }

    computeHorizontalZone(cx) {
        if (cx < 0.33) return "LEFT";
        if (cx > 0.67) return "RIGHT";
        return "CENTER";
    }
}
