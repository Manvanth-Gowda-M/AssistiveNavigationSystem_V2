/**
 * Speech manager: the Web Speech API plumbing around SpeechPolicy.
 *
 * Responsibilities kept here (and out of the policy, which stays pure):
 *   - priority queue with emergency pre-emption
 *   - cancelling an in-progress utterance when something urgent arrives
 *   - tracking when speech actually *finishes*, not when it is dequeued
 *   - working around the two Web Speech behaviours that break assistive apps:
 *     utterances that never fire `onend`, and Chrome's habit of pausing
 *     synthesis when the page is backgrounded
 *
 * The synthesiser is injected so this class is testable with a stub.
 */

import { SpeechConfig, SpeechPriority } from "../config/speechConfig.js";
import { SpeechPolicy, SpeechVerdict } from "./speechPolicy.js";
import { SpeechQueue, SpeechItem } from "./speechQueue.js";
import { VoiceManager } from "./voiceManager.js";

/** Hard ceiling on how long we wait for an `onend` that may never arrive. */
const UTTERANCE_WATCHDOG_MS = 6000;

export class SpeechManager {
    /**
     * @param {{synth?:SpeechSynthesis, voiceManager?:object, config?:object,
     *          now?:()=>number, onSpoken?:Function, onAnnounce?:Function}} [opts]
     */
    constructor({
        synth = (typeof window !== "undefined" ? window.speechSynthesis : null),
        voiceManager = null,
        config = SpeechConfig,
        now = () => Date.now(),
        onSpoken = () => {},
        onAnnounce = () => {}
    } = {}) {
        this.synth = synth;
        this.config = config;
        this.now = now;
        this.onSpoken = onSpoken;
        /** Mirror to an ARIA live region for screen-reader users. */
        this.onAnnounce = onAnnounce;

        this.voiceManager = voiceManager
            || (synth ? new VoiceManager() : { selectedVoice: null, rate: config.voice.rate, pitch: config.voice.pitch, volume: config.voice.volume });

        this.policy = new SpeechPolicy(config);
        this.queue = new SpeechQueue();

        this.enabled = true;
        this.isSpeaking = false;
        this._current = null;
        this._watchdogTimer = null;
        this.history = [];
        this.failures = 0;
    }

    get available() {
        return Boolean(this.synth);
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        if (!enabled) this.cancelAll();
    }

    /**
     * Submit a navigation decision. Most calls result in silence, which is the
     * intended behaviour.
     *
     * @param {object} decision
     * @param {{announceClear?:boolean}} [ctx]
     * @returns {{spoken:boolean, verdict:string, text?:string}}
     */
    speakDecision(decision, { announceClear = false } = {}) {
        const nowMs = this.now();
        const verdict = this.policy.evaluate({
            decision,
            nowMs,
            announceClear,
            enabled: this.enabled
        });

        if (!verdict.speak) {
            return { spoken: false, verdict: verdict.verdict };
        }

        this._dispatch({
            text: verdict.text,
            priority: verdict.priority,
            emergency: Boolean(verdict.emergency),
            interrupt: Boolean(verdict.interrupt),
            decision,
            nowMs
        });

        return { spoken: true, verdict: verdict.verdict, text: verdict.text };
    }

    /**
     * Speak a system/status message. Bypasses the navigation policy (these are
     * user-initiated or lifecycle events) but still respects the queue.
     */
    speakSystem(text, priority = SpeechPriority.SYSTEM, { interrupt = false } = {}) {
        if (!this.enabled || !text) return { spoken: false, verdict: SpeechVerdict.SILENT_DISABLED };
        this._dispatch({
            text,
            priority,
            emergency: priority <= SpeechPriority.IMMEDIATE_DANGER,
            interrupt,
            decision: null,
            nowMs: this.now()
        });
        return { spoken: true, verdict: SpeechVerdict.SPOKEN, text };
    }

    /** Longer descriptive speech, only on explicit user request. */
    speakDescription(text) {
        if (!this.enabled || !text) return { spoken: false };
        this._dispatch({
            text,
            priority: SpeechPriority.OBJECT,
            emergency: false,
            interrupt: false,
            decision: null,
            nowMs: this.now()
        });
        return { spoken: true, text };
    }

    _dispatch({ text, priority, emergency, interrupt, decision, nowMs }) {
        // An emergency cancels whatever is being said. Requirement 36: if we are
        // mid-way through "Path clear" when a hazard appears, the user must hear
        // "Stop." immediately, not after the reassurance finishes.
        if (interrupt || emergency) {
            this._cancelCurrent();
            this.queue.clear();
        }

        this.queue.enqueue(new SpeechItem({ text, priority, isEmergency: emergency }));
        this.policy.commit({ text, priority, emergency, decision, nowMs });

        this.history.push({ at: nowMs, text, priority, emergency });
        if (this.history.length > 60) this.history.shift();

        this.onAnnounce(text, emergency);
        this._pump();
    }

    _pump() {
        if (!this.synth || this.isSpeaking || this.queue.isEmpty()) return;

        const item = this.queue.dequeue();
        if (!item) return;

        let utterance;
        try {
            utterance = new SpeechSynthesisUtterance(item.text);
        } catch {
            this.failures += 1;
            return;
        }

        if (this.voiceManager.selectedVoice) utterance.voice = this.voiceManager.selectedVoice;
        utterance.rate = this.voiceManager.rate ?? this.config.voice.rate;
        utterance.pitch = this.voiceManager.pitch ?? this.config.voice.pitch;
        utterance.volume = this.voiceManager.volume ?? this.config.voice.volume;

        this.isSpeaking = true;
        this._current = utterance;

        const finish = (reason) => {
            if (this._current !== utterance) return;
            this._clearWatchdog();
            this.isSpeaking = false;
            this._current = null;
            this.onSpoken({ text: item.text, priority: item.priority, reason });
            // Small gap so consecutive instructions do not run together.
            setTimeout(() => this._pump(), 90);
        };

        utterance.onend = () => finish("end");
        utterance.onerror = (event) => {
            // `interrupted` and `canceled` are expected on the emergency path.
            if (event?.error && event.error !== "interrupted" && event.error !== "canceled") {
                this.failures += 1;
            }
            finish(event?.error || "error");
        };

        // Some mobile browsers drop `onend` entirely, which would wedge the
        // queue permanently. The watchdog guarantees forward progress.
        this._watchdogTimer = setTimeout(() => finish("watchdog"), UTTERANCE_WATCHDOG_MS);

        try {
            // Chrome pauses synthesis when the tab loses focus and does not
            // always resume it.
            if (this.synth.paused) this.synth.resume();
            this.synth.speak(utterance);
        } catch {
            this.failures += 1;
            finish("throw");
        }
    }

    _cancelCurrent() {
        this._clearWatchdog();
        this._current = null;
        this.isSpeaking = false;
        try {
            this.synth?.cancel();
        } catch { /* ignore */ }
    }

    _clearWatchdog() {
        if (this._watchdogTimer) {
            clearTimeout(this._watchdogTimer);
            this._watchdogTimer = null;
        }
    }

    cancelAll() {
        this._cancelCurrent();
        this.queue.clear();
    }

    /** Called on pause/stop so the next session starts from a clean slate. */
    reset() {
        this.cancelAll();
        this.policy.reset();
    }

    metrics() {
        return {
            available: this.available,
            enabled: this.enabled,
            speaking: this.isSpeaking,
            queued: this.queue.queue.length,
            failures: this.failures,
            ...this.policy.metrics()
        };
    }
}

export default SpeechManager;
