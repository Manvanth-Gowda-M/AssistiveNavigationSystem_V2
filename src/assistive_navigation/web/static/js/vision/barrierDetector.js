/**
 * Surface & Barrier Detector Module
 * Identifies large solid obstacles, walls, pillars, and closed doors that
 * lack explicit bounding boxes in generic object detection datasets.
 */

export class BarrierDetector {
    constructor() {
        this.sampleCanvas = document.createElement("canvas");
        this.sampleCanvas.width = 32;
        this.sampleCanvas.height = 24;
        this.ctx = this.sampleCanvas.getContext("2d", { willReadFrequently: true });
        this.consecutiveWallFrames = 0;
    }

    /**
     * Analyzes center field of view for large solid surface / wall obstruction.
     * Returns detection object or null.
     */
    detectWallBarrier(videoElement, timestamp = Date.now()) {
        if (!videoElement || videoElement.readyState < 2 || !videoElement.videoWidth) {
            return null;
        }

        try {
            const w = 32;
            const h = 24;
            this.ctx.drawImage(videoElement, 0, 0, w, h);
            const imgData = this.ctx.getImageData(0, 0, w, h);
            const data = imgData.data;

            // Analyze center corridor region: columns 8 to 24 (center 50%), rows 4 to 20 (middle 66%)
            let totalLuminance = 0;
            let centerCount = 0;
            const luminances = [];

            for (let y = 4; y < 20; y++) {
                for (let x = 8; x < 24; x++) {
                    const idx = (y * w + x) * 4;
                    const lum = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
                    luminances.push(lum);
                    totalLuminance += lum;
                    centerCount++;
                }
            }

            const meanLum = totalLuminance / centerCount;

            // Calculate standard deviation (texture roughness / contrast)
            let varianceSum = 0;
            for (let i = 0; i < luminances.length; i++) {
                const diff = luminances[i] - meanLum;
                varianceSum += diff * diff;
            }
            const stdDev = Math.sqrt(varianceSum / centerCount);

            // A wall / close barrier typically exhibits:
            // 1. Uniform color / low texture variance (stdDev < 16 on standard surfaces)
            // 2. Occupies the majority of the center field of view
            // 3. Not pitch black (meanLum > 22)
            const isSolidSurface = stdDev < 15.5 && meanLum > 22 && meanLum < 245;

            if (isSolidSurface) {
                this.consecutiveWallFrames++;
            } else {
                this.consecutiveWallFrames = Math.max(0, this.consecutiveWallFrames - 1);
            }

            // Confirm across 2 consecutive frames to avoid false positives during rapid camera pans
            if (this.consecutiveWallFrames >= 2) {
                return {
                    label: "wall",
                    confidence: 0.85,
                    boundingBox: [0.20, 0.15, 0.80, 0.90],
                    timestamp
                };
            }

            return null;
        } catch (err) {
            return null;
        }
    }
}
