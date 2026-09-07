/**
 * Adaptive frame scheduler.
 *
 * Two jobs:
 *   1. Decide *when* to run inference, and never let more than one run at a
 *      time. There is no queue: if inference is busy the frame is dropped and
 *      the next decision is made against a fresher frame (requirement 9).
 *   2. Continuously re-tune the inference rate from measured latency, so the
 *      operating point tracks the device instead of an arbitrary constant
 *      (requirements 10, 53).
 *
 * The controller steers **p95** latency, not the mean. A pipeline that averages
 * 40 ms but spikes to 400 ms every tenth frame feels broken while walking, and
 * a mean-based controller cannot see that.
 *
 * All time is passed in, so this module is deterministic and unit-testable.
 */

/** Fixed-capacity ring buffer with percentile queries. */
export class LatencyWindow {
    constructor(capacity = 60) {
        this.capacity = capacity;
        this._buf = new Float64Array(capacity);
        this._len = 0;
        this._head = 0;
        this._sorted = new Float64Array(capacity);
        this._sortedValid = false;
    }

    push(value) {
        if (!Number.isFinite(value)) return;
        this._buf[this._head] = value;
        this._head = (this._head + 1) % this.capacity;
        if (this._len < this.capacity) this._len += 1;
        this._sortedValid = false;
    }

    get length() {
        return this._len;
    }

    clear() {
        this._len = 0;
        this._head = 0;
        this._sortedValid = false;
    }

    _ensureSorted() {
        if (this._sortedValid) return;
        for (let i = 0; i < this._len; i += 1) this._sorted[i] = this._buf[i];
        // subarray().sort() sorts in place over just the populated region.
        this._sorted.subarray(0, this._len).sort();
        this._sortedValid = true;
    }

    percentile(p) {
        if (this._len === 0) return 0;
        this._ensureSorted();
        const idx = Math.min(this._len - 1, Math.max(0, Math.ceil((p / 100) * this._len) - 1));
        return this._sorted[idx];
    }

    mean() {
        if (this._len === 0) return 0;
        let sum = 0;
        for (let i = 0; i < this._len; i += 1) sum += this._buf[i];
        return sum / this._len;
    }

    max() {
        if (this._len === 0) return 0;
        this._ensureSorted();
        return this._sorted[this._len - 1];
    }
}

/** Reasons a frame was not submitted. Surfaced on the dashboard. */
export const DropReason = Object.freeze({
    BUSY: "busy",
    RATE_LIMIT: "rate-limit",
    NO_BUFFER: "no-buffer",
    NO_FRAME: "no-frame",
    QUALITY: "quality",
    PAUSED: "paused"
});

export class FrameScheduler {
    /**
     * @param {object} cfg VisionConfig.scheduler shape
     * @param {{targetFps?:number, maxFps?:number, minFps?:number}} [profile]
     */
    constructor(cfg, profile = {}) {
        this.cfg = cfg;
        this.minFps = profile.minFps ?? cfg.minFps;
        this.maxFps = profile.maxFps ?? cfg.maxFps;
        this.targetFps = profile.targetFps ?? cfg.startFps;
        this.ceilingFps = this.maxFps;

        this.intervalMs = 1000 / this.targetFps;

        this.latency = new LatencyWindow(cfg.latencyWindow);
        /** End-to-end: capture + inference + decode. What the user experiences. */
        this.endToEnd = new LatencyWindow(cfg.latencyWindow);

        this._lastSubmitTime = -Infinity;
        this._lastControlTime = 0;
        this._comfortableWindows = 0;

        this._inferenceTimestamps = [];
        this._cameraTimestamps = [];

        this.bestP95 = Infinity;
        this.thermalThrottled = false;
        this.adjustments = [];

        this.counters = {
            submitted: 0,
            completed: 0,
            failed: 0,
            dropped: 0,
            byReason: Object.create(null)
        };
    }

    /* --------------------------------------------------------------- gating */

    /**
     * @param {number} nowMs
     * @param {{busy:boolean, paused?:boolean, canCapture?:boolean}} state
     * @returns {{run:boolean, reason?:string}}
     */
    shouldRun(nowMs, state) {
        if (state.paused) return { run: false, reason: DropReason.PAUSED };
        if (state.busy) return { run: false, reason: DropReason.BUSY };
        if (state.canCapture === false) return { run: false, reason: DropReason.NO_BUFFER };
        if (nowMs - this._lastSubmitTime < this.intervalMs) {
            return { run: false, reason: DropReason.RATE_LIMIT };
        }
        return { run: true };
    }

    markSubmitted(nowMs) {
        this._lastSubmitTime = nowMs;
        this.counters.submitted += 1;
    }

    /**
     * @param {number} nowMs
     * @param {number} inferenceLatencyMs adapter-reported latency
     * @param {number} [endToEndMs] capture + inference + decode
     */
    markCompleted(nowMs, inferenceLatencyMs, endToEndMs = null) {
        this.counters.completed += 1;
        this.latency.push(inferenceLatencyMs);
        this.endToEnd.push(endToEndMs ?? inferenceLatencyMs);

        this._inferenceTimestamps.push(nowMs);
        this._trimWindow(this._inferenceTimestamps, nowMs);
    }

    markFailed() {
        this.counters.failed += 1;
    }

    /**
     * Record a frame that was not submitted.
     *
     * `RATE_LIMIT` is intentionally *not* counted as a dropped frame: skipping
     * because we are deliberately running at 12 FPS on a 30 FPS camera is
     * correct behaviour, not a symptom. Only contention counts.
     */
    markDropped(reason) {
        this.counters.byReason[reason] = (this.counters.byReason[reason] || 0) + 1;
        if (reason === DropReason.BUSY || reason === DropReason.NO_BUFFER) {
            this.counters.dropped += 1;
        }
    }

    /** Called once per rendered frame to measure the camera's real cadence. */
    tickCamera(nowMs) {
        this._cameraTimestamps.push(nowMs);
        this._trimWindow(this._cameraTimestamps, nowMs);
    }

    _trimWindow(list, nowMs, windowMs = 2000) {
        const cutoff = nowMs - windowMs;
        while (list.length && list[0] < cutoff) list.shift();
    }

    /* ------------------------------------------------------------ controller */

    /**
     * Re-tune the inference rate. Call frequently; it self-throttles to
     * `controlIntervalMs`.
     *
     * @param {number} nowMs
     * @returns {{changed:boolean, targetFps:number, action:string, p95:number}|null}
     */
    control(nowMs) {
        if (nowMs - this._lastControlTime < this.cfg.controlIntervalMs) return null;
        this._lastControlTime = nowMs;

        // Need a meaningful sample before moving the operating point.
        if (this.latency.length < 6) return null;

        const p95 = Math.round(this.latency.percentile(95));
        if (p95 > 0 && p95 < this.bestP95) this.bestP95 = p95;

        // Sustained-load / thermal guard: compare against the best this session
        // has ever achieved, not against a fixed constant. A device that started
        // at 45 ms and is now at 90 ms is throttling even though 90 ms is
        // nominally inside budget.
        const regression = this.bestP95 > 0 && Number.isFinite(this.bestP95)
            ? p95 / this.bestP95
            : 1;
        const throttling = regression >= this.cfg.thermalRegressionRatio;

        let action = "hold";
        const previous = this.targetFps;

        if (p95 > this.cfg.maxP95LatencyMs || throttling) {
            // Shed load immediately and decisively; recovering slowly is fine,
            // collapsing is not.
            this._comfortableWindows = 0;
            this.thermalThrottled = throttling;
            this.targetFps = Math.max(this.minFps, Math.round(this.targetFps * 0.7));
            // Lower the ceiling too, so the ramp-up logic cannot immediately
            // undo the correction.
            this.ceilingFps = Math.max(this.minFps, Math.min(this.ceilingFps, this.targetFps + 2));
            action = throttling ? "shed-thermal" : "shed-latency";
        } else if (p95 > this.cfg.targetP95LatencyMs) {
            this._comfortableWindows = 0;
            this.targetFps = Math.max(this.minFps, this.targetFps - 1);
            action = "ease-down";
        } else {
            this._comfortableWindows += 1;
            if (this._comfortableWindows >= this.cfg.rampUpWindows && this.targetFps < this.ceilingFps) {
                this._comfortableWindows = 0;
                this.targetFps = Math.min(this.ceilingFps, this.targetFps + 1);
                action = "ramp-up";
            }
            if (this.thermalThrottled && this.targetFps >= this.ceilingFps) {
                this.thermalThrottled = false;
            }
        }

        this.intervalMs = 1000 / this.targetFps;
        const changed = this.targetFps !== previous;

        if (changed) {
            this.adjustments.push({ at: Math.round(nowMs), action, from: previous, to: this.targetFps, p95 });
            if (this.adjustments.length > 40) this.adjustments.shift();
        }

        return { changed, targetFps: this.targetFps, action, p95 };
    }

    /**
     * Lower the ceiling permanently for this session. Used by the degradation
     * controller when it disables an expensive stage but still sees pressure.
     */
    clampCeiling(fps) {
        this.ceilingFps = Math.max(this.minFps, Math.min(this.ceilingFps, fps));
        if (this.targetFps > this.ceilingFps) {
            this.targetFps = this.ceilingFps;
            this.intervalMs = 1000 / this.targetFps;
        }
    }

    /* --------------------------------------------------------------- metrics */

    get inferenceFps() {
        return this._rate(this._inferenceTimestamps);
    }

    get cameraFps() {
        return this._rate(this._cameraTimestamps);
    }

    _rate(list) {
        if (list.length < 2) return 0;
        const span = (list[list.length - 1] - list[0]) / 1000;
        if (span <= 0) return 0;
        return (list.length - 1) / span;
    }

    metrics() {
        return {
            targetFps: this.targetFps,
            ceilingFps: this.ceilingFps,
            intervalMs: Math.round(this.intervalMs),
            inferenceFps: Number(this.inferenceFps.toFixed(1)),
            cameraFps: Number(this.cameraFps.toFixed(1)),
            latencyP50: Math.round(this.latency.percentile(50)),
            latencyP95: Math.round(this.latency.percentile(95)),
            latencyMean: Math.round(this.latency.mean()),
            latencyMax: Math.round(this.latency.max()),
            endToEndP50: Math.round(this.endToEnd.percentile(50)),
            endToEndP95: Math.round(this.endToEnd.percentile(95)),
            bestP95: Number.isFinite(this.bestP95) ? this.bestP95 : null,
            thermalThrottled: this.thermalThrottled,
            submitted: this.counters.submitted,
            completed: this.counters.completed,
            failed: this.counters.failed,
            droppedFrames: this.counters.dropped,
            dropsByReason: { ...this.counters.byReason },
            recentAdjustments: this.adjustments.slice(-5)
        };
    }

    reset() {
        this.latency.clear();
        this.endToEnd.clear();
        this._inferenceTimestamps.length = 0;
        this._cameraTimestamps.length = 0;
        this._comfortableWindows = 0;
        this._lastSubmitTime = -Infinity;
        this.counters.submitted = 0;
        this.counters.completed = 0;
        this.counters.failed = 0;
        this.counters.dropped = 0;
        this.counters.byReason = Object.create(null);
    }
}
