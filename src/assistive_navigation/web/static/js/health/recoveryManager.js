/**
 * Watchdog and recovery ladder.
 *
 * Requirement 17 and 48: when inference stops, fix it without making the user
 * reload the page, and if it cannot be fixed, say so plainly and stop pretending
 * to guide them.
 *
 * The ladder, in order:
 *   1. RETRY            - the pipeline may just have hiccuped. Cheapest step.
 *   2. RESTART_WORKER   - rebuild the inference worker and session in place.
 *   3. FALLBACK_BACKEND - WebGPU -> WASM SIMD -> WASM -> main-thread runtime.
 *   4. RELOAD_MODEL     - re-fetch weights with the IndexedDB entry dropped,
 *                         in case the cached bytes are corrupt.
 *   5. REINITIALISE     - full re-selection from the top of the backend plan.
 *
 * Each step is attempted at most once per escalation cycle, with backoff. A
 * successful inference resets the ladder to the bottom, because the next failure
 * may be unrelated and should get the cheap treatment first.
 */

import { VisionConfig } from "../config/visionConfig.js";

export const RecoveryStep = Object.freeze({
    RETRY: "RETRY",
    RESTART_WORKER: "RESTART_WORKER",
    FALLBACK_BACKEND: "FALLBACK_BACKEND",
    RELOAD_MODEL: "RELOAD_MODEL",
    REINITIALISE: "REINITIALISE",
    EXHAUSTED: "EXHAUSTED"
});

const LADDER = [
    RecoveryStep.RETRY,
    RecoveryStep.RESTART_WORKER,
    RecoveryStep.FALLBACK_BACKEND,
    RecoveryStep.RELOAD_MODEL,
    RecoveryStep.REINITIALISE
];

/**
 * Timer-based stall detector.
 *
 * Deliberately independent of the render loop: if the loop itself has stopped
 * (the failure mode that matters most) a loop-driven check would never fire.
 */
export class VisionWatchdog {
    /**
     * @param {{monitor:object, onStall:(detail:object)=>void, cfg?:object,
     *          intervalMs?:number, now?:()=>number}} opts
     */
    constructor({ monitor, onStall, cfg = VisionConfig.health, intervalMs = 500, now = () => performance.now() }) {
        this.monitor = monitor;
        this.onStall = onStall;
        this.cfg = cfg;
        this.intervalMs = intervalMs;
        this.now = now;

        this._timer = null;
        this.enabled = false;
        this.stallsDetected = 0;
        this._suppressUntilMs = 0;
    }

    start() {
        if (this._timer) return;
        this.enabled = true;
        this._timer = setInterval(() => this._tick(), this.intervalMs);
    }

    stop() {
        this.enabled = false;
        if (this._timer) {
            clearInterval(this._timer);
            this._timer = null;
        }
    }

    /** Hold off while a recovery attempt is already running. */
    suppressFor(ms) {
        this._suppressUntilMs = this.now() + ms;
    }

    _tick() {
        if (!this.enabled) return;
        const nowMs = this.now();
        if (nowMs < this._suppressUntilMs) return;

        const silence = this.monitor.silenceMs();
        if (silence <= this.cfg.watchdogTimeoutMs) return;

        this.stallsDetected += 1;
        // Avoid re-firing while the handler works; the handler extends this.
        this.suppressFor(this.cfg.watchdogTimeoutMs);
        this.onStall({
            silenceMs: Math.round(silence),
            consecutiveFailures: this.monitor.consecutiveInferenceFailures,
            stallsDetected: this.stallsDetected
        });
    }
}

export class RecoveryManager {
    /**
     * @param {object} opts
     * @param {object} opts.monitor VisionHealthMonitor
     * @param {object} opts.handlers one async function per RecoveryStep, each
     *        returning `{ok:boolean, detail?:any}`
     * @param {(state:{step:string, attempt:number})=>void} [opts.onProgress]
     * @param {(detail:object)=>void} [opts.onExhausted]
     */
    constructor({ monitor, handlers, cfg = VisionConfig.health, onProgress = () => {}, onExhausted = () => {} }) {
        this.monitor = monitor;
        this.handlers = handlers;
        this.cfg = cfg;
        this.onProgress = onProgress;
        this.onExhausted = onExhausted;

        this.ladderIndex = 0;
        this.cycleAttempts = 0;
        this.inProgress = false;
        this.lastStep = null;
        this.lastResult = null;
        this.history = [];
    }

    get exhausted() {
        return this.ladderIndex >= LADDER.length || this.cycleAttempts >= this.cfg.maxRecoveryAttempts;
    }

    /** A healthy inference resets the ladder. */
    notifySuccess() {
        if (this.ladderIndex === 0 && this.cycleAttempts === 0) return;
        this.ladderIndex = 0;
        this.cycleAttempts = 0;
    }

    /**
     * Run the next recovery step.
     * @returns {Promise<{ok:boolean, step:string, exhausted:boolean, detail?:any}>}
     */
    async escalate(context = {}) {
        if (this.inProgress) {
            return { ok: false, step: this.lastStep, exhausted: this.exhausted, detail: "already recovering" };
        }
        if (this.exhausted) {
            this.monitor.markDead("recovery ladder exhausted");
            this.onExhausted({ attempts: this.cycleAttempts, context });
            return { ok: false, step: RecoveryStep.EXHAUSTED, exhausted: true };
        }

        this.inProgress = true;
        const step = LADDER[this.ladderIndex];
        this.lastStep = step;
        this.cycleAttempts += 1;
        this.monitor.markRecoveryAttempt(step);
        this.onProgress({ step, attempt: this.cycleAttempts });

        const backoffIndex = Math.min(this.cfg.recoveryBackoffMs.length - 1, this.cycleAttempts - 1);
        const backoff = this.cfg.recoveryBackoffMs[backoffIndex];

        let result = { ok: false, detail: "no handler" };
        try {
            if (backoff > 0) await new Promise((r) => setTimeout(r, backoff));
            const handler = this.handlers[step];
            if (typeof handler === "function") {
                result = await handler(context);
            }
        } catch (err) {
            result = { ok: false, detail: String(err?.message || err) };
        }

        this.lastResult = result;
        this.history.push({ step, ok: Boolean(result.ok), detail: result.detail ?? null, at: Date.now() });
        if (this.history.length > 30) this.history.shift();

        if (result.ok) {
            this.monitor.markRecovered(step);
            // Do NOT reset the ladder here. Recovery "worked" means the step
            // completed, not that the pipeline is producing results again. The
            // ladder resets on the next successful inference, via
            // notifySuccess().
        } else {
            this.ladderIndex += 1;
        }

        this.inProgress = false;

        if (!result.ok && this.exhausted) {
            this.monitor.markDead("recovery ladder exhausted");
            this.onExhausted({ attempts: this.cycleAttempts, lastStep: step, context });
        }

        return { ok: Boolean(result.ok), step, exhausted: this.exhausted, detail: result.detail };
    }

    reset() {
        this.ladderIndex = 0;
        this.cycleAttempts = 0;
        this.inProgress = false;
    }

    metrics() {
        return {
            ladderIndex: this.ladderIndex,
            nextStep: LADDER[this.ladderIndex] || RecoveryStep.EXHAUSTED,
            cycleAttempts: this.cycleAttempts,
            inProgress: this.inProgress,
            lastStep: this.lastStep,
            exhausted: this.exhausted,
            history: this.history.slice(-5)
        };
    }
}

export { LADDER as RECOVERY_LADDER };
