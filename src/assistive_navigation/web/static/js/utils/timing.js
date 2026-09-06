/**
 * Timing, FPS calculation, and performance monitor utilities.
 */

export class PerformanceTracker {
    constructor(windowSize = 20) {
        this.windowSize = windowSize;
        this.timestamps = [];
        this.inferenceLatencies = [];
        this.lastFrameTime = performance.now();
    }

    tick() {
        const now = performance.now();
        this.timestamps.push(now);
        if (this.timestamps.length > this.windowSize) {
            this.timestamps.shift();
        }
        this.lastFrameTime = now;
    }

    recordInference(latencyMs) {
        this.inferenceLatencies.push(latencyMs);
        if (this.inferenceLatencies.length > this.windowSize) {
            this.inferenceLatencies.shift();
        }
    }

    getFps() {
        if (this.timestamps.length < 2) return 0;
        const delta = (this.timestamps[this.timestamps.length - 1] - this.timestamps[0]) / 1000;
        if (delta <= 0) return 0;
        return Math.round((this.timestamps.length - 1) / delta);
    }

    getAvgLatency() {
        if (this.inferenceLatencies.length === 0) return 0;
        const sum = this.inferenceLatencies.reduce((a, b) => a + b, 0);
        return Math.round(sum / this.inferenceLatencies.length);
    }
}
