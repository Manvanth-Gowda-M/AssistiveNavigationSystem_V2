/**
 * Exponential moving average and velocity smoothing utilities.
 */

export class SmoothingUtils {
    /**
     * Exponential moving average.
     */
    static expSmooth(currentVal, previousVal, alpha = 0.4) {
        if (previousVal === null || previousVal === undefined) return currentVal;
        return alpha * currentVal + (1 - alpha) * previousVal;
    }

    /**
     * Smooth an array of bounding box coordinates [x1, y1, x2, y2].
     */
    static smoothBox(currentBox, prevBox, alpha = 0.5) {
        if (!prevBox) return currentBox;
        return [
            this.expSmooth(currentBox[0], prevBox[0], alpha),
            this.expSmooth(currentBox[1], prevBox[1], alpha),
            this.expSmooth(currentBox[2], prevBox[2], alpha),
            this.expSmooth(currentBox[3], prevBox[3], alpha)
        ];
    }

    /**
     * Compute approach rate based on bounding box area expansion over time delta.
     * Positive = Approaching / Expanding
     * Negative = Receding / Shrinking
     */
    static computeAreaGrowthRate(currentArea, prevArea, dtSeconds) {
        if (!prevArea || dtSeconds <= 0) return 0;
        return (currentArea - prevArea) / dtSeconds;
    }
}
