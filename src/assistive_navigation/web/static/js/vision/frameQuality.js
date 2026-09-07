/**
 * Frame quality estimation and short-term degradation tolerance.
 *
 * Walking breaks cameras in predictable ways: motion blur on every footfall,
 * exposure hunting when you step from shade into sun, and the odd frame where
 * a hand covers the lens. The correct response to one bad frame is to keep
 * going; the correct response to twenty is to stop giving confident directions
 * (requirements 39-41).
 *
 * So this module produces a *rolling* score and a three-state verdict, never a
 * per-frame pass/fail.
 *
 * Operates on an 8-bit luminance plane, which keeps it cheap and lets it be
 * unit-tested with plain arrays.
 */

import { VisionConfig } from "../config/visionConfig.js";

export const QualityState = Object.freeze({
    GOOD: "GOOD",
    DEGRADED: "DEGRADED",
    UNUSABLE: "UNUSABLE"
});

export const QualityIssue = Object.freeze({
    DARK: "dark",
    BRIGHT: "bright",
    BLUR: "blur",
    LOW_CONTRAST: "low-contrast"
});

/**
 * Single-frame measurements from a luminance plane.
 *
 * `blur` is normalised 4-neighbour Laplacian energy. Measured on the analysis
 * thumbnail rather than the full frame, so it detects *gross* smear (the kind
 * that actually destroys detection) rather than fine defocus. That is the right
 * trade: it costs ~3k operations instead of ~300k.
 *
 * @param {Uint8Array|number[]} luma
 * @param {number} width
 * @param {number} height
 * @returns {{brightness:number, contrast:number, blur:number}}
 */
export function analyseLuminance(luma, width, height) {
    const n = width * height;
    if (!n || !luma || luma.length < n) return { brightness: 0, contrast: 0, blur: 0 };

    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += luma[i];
    const brightness = sum / n;

    let varianceSum = 0;
    for (let i = 0; i < n; i += 1) {
        const d = luma[i] - brightness;
        varianceSum += d * d;
    }
    const contrast = Math.sqrt(varianceSum / n);

    let energy = 0;
    let samples = 0;
    for (let y = 1; y < height - 1; y += 1) {
        const row = y * width;
        for (let x = 1; x < width - 1; x += 1) {
            const c = luma[row + x];
            const lap = 4 * c - luma[row + x - 1] - luma[row + x + 1] - luma[row - width + x] - luma[row + width + x];
            energy += lap < 0 ? -lap : lap;
            samples += 1;
        }
    }
    // Divide by 4*255 so a maximally alternating pattern maps to ~1.
    const blur = samples > 0 ? energy / (samples * 1020) : 0;

    return { brightness, contrast, blur };
}

/**
 * Convert measurements into a 0..1 usability score plus the issues found.
 *
 * The score is the product of three sub-scores rather than an average: a frame
 * that is pitch black should not be rescued by having good contrast in the
 * noise.
 *
 * @returns {{score:number, issues:string[]}}
 */
export function scoreFrame({ brightness, contrast, blur }, cfg = VisionConfig.quality) {
    const issues = [];

    // Exposure: 1.0 in the comfortable band, tapering to 0 at the limits.
    let exposureScore;
    if (brightness < cfg.darkThreshold) {
        issues.push(QualityIssue.DARK);
        exposureScore = Math.max(0, brightness / cfg.darkThreshold) * 0.6;
    } else if (brightness > cfg.brightThreshold) {
        issues.push(QualityIssue.BRIGHT);
        exposureScore = Math.max(0, (255 - brightness) / (255 - cfg.brightThreshold)) * 0.6;
    } else {
        const lower = cfg.darkThreshold;
        const upper = cfg.brightThreshold;
        const mid = (lower + upper) / 2;
        const halfSpan = (upper - lower) / 2;
        exposureScore = 1 - 0.25 * Math.min(1, Math.abs(brightness - mid) / halfSpan);
    }

    let blurScore;
    if (blur < cfg.blurThreshold) {
        issues.push(QualityIssue.BLUR);
        blurScore = Math.max(0.15, blur / cfg.blurThreshold) * 0.7;
    } else {
        blurScore = Math.min(1, 0.7 + 0.3 * (blur / (cfg.blurThreshold * 3)));
    }

    let contrastScore;
    if (contrast < cfg.lowContrastThreshold) {
        issues.push(QualityIssue.LOW_CONTRAST);
        contrastScore = Math.max(0.2, contrast / cfg.lowContrastThreshold) * 0.75;
    } else {
        contrastScore = 1;
    }

    const score = Math.max(0, Math.min(1, exposureScore * blurScore * contrastScore));
    return { score, issues };
}

export class FrameQualityMonitor {
    constructor(cfg = VisionConfig.quality) {
        this.cfg = cfg;
        this._scores = [];
        this._poorRun = 0;
        this._goodRun = 0;
        this.state = QualityState.GOOD;
        this.lastMeasurement = { brightness: 128, contrast: 40, blur: 0.05 };
        this.lastIssues = [];
        this.samples = 0;
        /** Counts, for the dashboard: which failure mode dominates. */
        this.issueCounts = Object.create(null);
    }

    /**
     * @param {Uint8Array} luma
     * @returns {{state:string, score:number, issues:string[], rollingScore:number}}
     */
    update(luma, width, height) {
        const measurement = analyseLuminance(luma, width, height);
        const { score, issues } = scoreFrame(measurement, this.cfg);

        this.lastMeasurement = measurement;
        this.lastIssues = issues;
        this.samples += 1;
        for (const issue of issues) {
            this.issueCounts[issue] = (this.issueCounts[issue] || 0) + 1;
        }

        this._scores.push(score);
        if (this._scores.length > this.cfg.window) this._scores.shift();

        const rolling = this.rollingScore;

        // Hysteresis in both directions: entering and leaving a degraded state
        // both require sustained evidence, so a single footfall cannot flip it.
        const poor = score < this.cfg.minScoreForDirection;
        if (poor) {
            this._poorRun += 1;
            this._goodRun = 0;
        } else {
            this._goodRun += 1;
            this._poorRun = 0;
        }

        if (rolling < this.cfg.minScoreForOperation && this._poorRun >= this.cfg.degradeAfterPoorFrames) {
            this.state = QualityState.UNUSABLE;
        } else if (this._poorRun >= this.cfg.degradeAfterPoorFrames) {
            this.state = QualityState.DEGRADED;
        } else if (this._goodRun >= this.cfg.recoverAfterGoodFrames) {
            this.state = QualityState.GOOD;
        }

        return { state: this.state, score, issues, rollingScore: rolling };
    }

    get rollingScore() {
        if (this._scores.length === 0) return 1;
        let sum = 0;
        for (const s of this._scores) sum += s;
        return sum / this._scores.length;
    }

    /** True when perception is good enough to name a direction. */
    get allowsDirectionalGuidance() {
        return this.state === QualityState.GOOD
            || (this.state === QualityState.DEGRADED && this.rollingScore >= this.cfg.minScoreForDirection);
    }

    /** True when perception is too unreliable to operate at all. */
    get requiresStop() {
        return this.state === QualityState.UNUSABLE;
    }

    /** Dominant issue label, for speech selection. */
    get dominantIssue() {
        let best = null;
        let bestCount = 0;
        for (const [issue, count] of Object.entries(this.issueCounts)) {
            if (count > bestCount) {
                best = issue;
                bestCount = count;
            }
        }
        return best;
    }

    metrics() {
        return {
            state: this.state,
            score: Number(this.rollingScore.toFixed(3)),
            brightness: Math.round(this.lastMeasurement.brightness),
            contrast: Math.round(this.lastMeasurement.contrast),
            blur: Number(this.lastMeasurement.blur.toFixed(4)),
            issues: this.lastIssues.slice(),
            dominantIssue: this.dominantIssue,
            samples: this.samples,
            allowsDirection: this.allowsDirectionalGuidance
        };
    }

    reset() {
        this._scores.length = 0;
        this._poorRun = 0;
        this._goodRun = 0;
        this.state = QualityState.GOOD;
        this.issueCounts = Object.create(null);
        this.samples = 0;
    }
}
