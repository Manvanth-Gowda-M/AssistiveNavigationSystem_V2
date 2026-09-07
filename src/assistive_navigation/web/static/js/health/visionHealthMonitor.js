/**
 * Vision health monitor.
 *
 * The pre-upgrade build had no way to know it had stopped working. Its detector
 * swallowed every error and returned the previous frame's results forever, so a
 * lost WebGL context looked exactly like a static scene. That is the single
 * worst failure mode an assistive system can have: confidently silent.
 *
 * This module makes "is the pipeline actually alive?" an explicit, observable
 * fact.
 *
 * Pure bookkeeping plus an optional `performance.memory` sampler. No DOM.
 */

import { VisionConfig } from "../config/visionConfig.js";

export const HealthState = Object.freeze({
    STARTING: "STARTING",
    HEALTHY: "HEALTHY",
    /** Working, but something measurable is wrong. */
    DEGRADED: "DEGRADED",
    /** Not producing results; recovery is in progress or due. */
    FAILING: "FAILING",
    /** Recovery exhausted. */
    DEAD: "DEAD"
});

export class VisionHealthMonitor {
    /**
     * @param {{cfg?:object, now?:()=>number}} [opts]
     */
    constructor({ cfg = VisionConfig.health, now = () => performance.now() } = {}) {
        this.cfg = cfg;
        this.now = now;

        this.state = HealthState.STARTING;
        this.startedAtMs = now();

        this.lastSuccessfulInferenceMs = 0;
        this.lastFailureMs = 0;
        this.lastFailureMessage = null;
        this.consecutiveInferenceFailures = 0;
        this.totalInferences = 0;
        this.totalFailures = 0;

        this.modelStatus = "unloaded";
        this.backend = null;
        this.workerStatus = "none";
        this.trackingCount = 0;
        this.degradationLevel = 1;

        this.memoryWarnings = 0;
        this._heapSamples = [];
        this._lastHeapSampleMs = 0;
        this.heapBaselineBytes = null;
        this.heapLatestBytes = null;
        this.heapGrowthBytes = 0;

        this.recoveryAttempts = 0;
        this.recoveries = 0;
        this.events = [];
    }

    /* -------------------------------------------------------------- events */

    record(kind, detail = {}) {
        this.events.push({ at: Math.round(this.now()), kind, ...detail });
        if (this.events.length > 120) this.events.shift();
    }

    markModelStatus(status, backend = this.backend) {
        this.modelStatus = status;
        this.backend = backend;
        this.record("model-status", { status, backend });
    }

    markWorkerStatus(status) {
        this.workerStatus = status;
        this.record("worker-status", { status });
    }

    markInferenceSuccess() {
        this.lastSuccessfulInferenceMs = this.now();
        this.consecutiveInferenceFailures = 0;
        this.totalInferences += 1;
        if (this.state === HealthState.STARTING || this.state === HealthState.FAILING) {
            this.state = HealthState.HEALTHY;
        }
    }

    markInferenceFailure(message) {
        this.lastFailureMs = this.now();
        this.lastFailureMessage = message;
        this.consecutiveInferenceFailures += 1;
        this.totalFailures += 1;
        this.record("inference-failure", { message, consecutive: this.consecutiveInferenceFailures });

        if (this.consecutiveInferenceFailures >= this.cfg.maxConsecutiveFailures) {
            this.state = HealthState.FAILING;
        }
    }

    markRecoveryAttempt(step) {
        this.recoveryAttempts += 1;
        this.state = HealthState.FAILING;
        this.record("recovery-attempt", { step, attempt: this.recoveryAttempts });
    }

    markRecovered(step) {
        this.recoveries += 1;
        this.consecutiveInferenceFailures = 0;
        this.state = HealthState.HEALTHY;
        this.record("recovered", { step });
    }

    markDead(reason) {
        this.state = HealthState.DEAD;
        this.record("dead", { reason });
    }

    setDegradationLevel(level) {
        if (level === this.degradationLevel) return;
        this.degradationLevel = level;
        this.state = level >= 2 && this.state === HealthState.HEALTHY ? HealthState.DEGRADED : this.state;
        this.record("degradation", { level });
    }

    setTrackingCount(n) {
        this.trackingCount = n;
    }

    /* -------------------------------------------------------------- memory */

    /**
     * Sample the JS heap. Chromium-only; absent elsewhere, which is fine - the
     * absence is reported rather than faked.
     *
     * A steadily climbing heap over a ten-minute walk is the signature of the
     * leak class this architecture is designed to avoid, so it is worth the
     * three lines it costs to watch for it.
     */
    sampleMemory(perf = (typeof performance !== "undefined" ? performance : null)) {
        const nowMs = this.now();
        if (nowMs - this._lastHeapSampleMs < this.cfg.heapSampleIntervalMs) return null;
        this._lastHeapSampleMs = nowMs;

        const mem = perf?.memory;
        if (!mem || !Number.isFinite(mem.usedJSHeapSize)) return null;

        const used = mem.usedJSHeapSize;
        this.heapLatestBytes = used;
        if (this.heapBaselineBytes === null) this.heapBaselineBytes = used;

        this._heapSamples.push({ at: nowMs, used });
        if (this._heapSamples.length > this.cfg.heapSampleWindow) this._heapSamples.shift();

        // Compare against the minimum observed, not the first sample: startup
        // allocates and then settles, and we care about growth after settling.
        let minUsed = used;
        for (const s of this._heapSamples) if (s.used < minUsed) minUsed = s.used;
        this.heapGrowthBytes = used - minUsed;

        if (this.heapGrowthBytes > this.cfg.heapGrowthWarningBytes) {
            this.memoryWarnings += 1;
            this.record("memory-warning", {
                growthMb: Math.round(this.heapGrowthBytes / 1e6),
                usedMb: Math.round(used / 1e6)
            });
            // Re-baseline so one sustained climb does not emit a warning per
            // sample forever.
            this._heapSamples.length = 0;
            this._heapSamples.push({ at: nowMs, used });
        }

        return {
            usedMb: Math.round(used / 1e6),
            growthMb: Math.round(this.heapGrowthBytes / 1e6),
            limitMb: Number.isFinite(mem.jsHeapSizeLimit) ? Math.round(mem.jsHeapSizeLimit / 1e6) : null
        };
    }

    /* -------------------------------------------------------------- status */

    /** Milliseconds since the last successful inference. */
    silenceMs() {
        if (this.lastSuccessfulInferenceMs === 0) return this.now() - this.startedAtMs;
        return this.now() - this.lastSuccessfulInferenceMs;
    }

    /** True when the pipeline has gone quiet for longer than the watchdog allows. */
    get isStalled() {
        return this.silenceMs() > this.cfg.watchdogTimeoutMs;
    }

    get isOperational() {
        return this.state === HealthState.HEALTHY || this.state === HealthState.DEGRADED;
    }

    snapshot() {
        return {
            state: this.state,
            modelStatus: this.modelStatus,
            backend: this.backend,
            workerStatus: this.workerStatus,
            degradationLevel: this.degradationLevel,
            trackingCount: this.trackingCount,
            lastSuccessfulInferenceMs: Math.round(this.lastSuccessfulInferenceMs),
            silenceMs: Math.round(this.silenceMs()),
            consecutiveInferenceFailures: this.consecutiveInferenceFailures,
            totalInferences: this.totalInferences,
            totalFailures: this.totalFailures,
            lastFailureMessage: this.lastFailureMessage,
            recoveryAttempts: this.recoveryAttempts,
            recoveries: this.recoveries,
            memoryWarnings: this.memoryWarnings,
            heapUsedMb: this.heapLatestBytes !== null ? Math.round(this.heapLatestBytes / 1e6) : null,
            heapGrowthMb: Math.round(this.heapGrowthBytes / 1e6),
            uptimeSeconds: Math.round((this.now() - this.startedAtMs) / 1000)
        };
    }

    resetSession() {
        this.startedAtMs = this.now();
        this.lastSuccessfulInferenceMs = 0;
        this.consecutiveInferenceFailures = 0;
        this._heapSamples.length = 0;
        this.heapBaselineBytes = null;
        this.heapGrowthBytes = 0;
        this.state = HealthState.STARTING;
    }
}

export default VisionHealthMonitor;
