/**
 * Speech policy: what to say, and whether to say anything at all.
 *
 * Separated from the synthesis plumbing so it can be unit-tested without a
 * browser. This is where "the system speaks too much" is fixed.
 *
 * The policy answers one question per decision: **has the situation changed in a
 * way the user needs to hear?** Not "is there an obstacle" - the user already
 * knows, we told them. Concretely it gates on:
 *
 *   - the action changing,
 *   - the risk changing by a material amount,
 *   - the scene signature changing (different object, different distance band),
 *   - a per-priority cooldown,
 *   - a slow reminder for a still-live instruction,
 *   - and an unconditional emergency bypass.
 *
 * Silence is the default return value.
 */

import { SpeechConfig, SpeechPriority } from "../config/speechConfig.js";
import { Action, Urgency } from "../navigation/decisionEngine.js";
import { HazardTier, tierForLabel } from "../config/classCatalog.js";

/** Why the policy chose to speak or stay silent. Surfaced on the dashboard. */
export const SpeechVerdict = Object.freeze({
    SPOKEN: "spoken",
    SILENT_NO_CHANGE: "silent-no-change",
    SILENT_COOLDOWN: "silent-cooldown",
    SILENT_CLEAR_PATH: "silent-clear-path",
    SILENT_DISABLED: "silent-disabled",
    SILENT_RATE_LIMIT: "silent-rate-limit",
    EMERGENCY_BYPASS: "emergency-bypass"
});

/**
 * Map a decision to its phrase.
 *
 * Directional phrases carry no object description at all. That is deliberate:
 * "Move slightly left" is actionable in under a second, and appending "chair on
 * your right" both delays the action and competes with the next instruction.
 * Object identity is available on the screen and via the describe-scene action
 * for users who want it.
 *
 * @param {object} decision
 * @returns {{text:string, priority:number}|null} null means say nothing
 */
export function phraseForDecision(decision, phrases = SpeechConfig.phrases) {
    const obstacle = decision.primaryObstacle;
    const tier = obstacle ? tierForLabel(obstacle.label) : null;

    switch (decision.action) {
        case Action.STOP: {
            if (decision.reason === "perception_unreliable") {
                return { text: phrases.STOP_UNRELIABLE, priority: SpeechPriority.STOP };
            }
            if (tier === HazardTier.VEHICLE) {
                return { text: phrases.STOP_VEHICLE, priority: SpeechPriority.STOP };
            }
            if (decision.reason === "rapid_approach") {
                return { text: phrases.STOP_APPROACHING, priority: SpeechPriority.STOP };
            }
            if (decision.reason === "obstacle_very_near") {
                return { text: phrases.STOP_VERY_NEAR, priority: SpeechPriority.STOP };
            }
            if (decision.reason === "blocked_all_directions"
                || decision.reason === "blocked_both_sides"
                || decision.reason === "no_passable_gap") {
                return { text: phrases.STOP_BLOCKED, priority: SpeechPriority.STOP };
            }
            return { text: phrases.STOP, priority: SpeechPriority.STOP };
        }

        case Action.MOVE_LEFT:
            return { text: phrases.MOVE_LEFT, priority: SpeechPriority.DIRECTION };
        case Action.MOVE_SLIGHTLY_LEFT:
            return { text: phrases.MOVE_SLIGHTLY_LEFT, priority: SpeechPriority.DIRECTION };
        case Action.MOVE_RIGHT:
            return { text: phrases.MOVE_RIGHT, priority: SpeechPriority.DIRECTION };
        case Action.MOVE_SLIGHTLY_RIGHT:
            return { text: phrases.MOVE_SLIGHTLY_RIGHT, priority: SpeechPriority.DIRECTION };

        case Action.CAUTION: {
            if (decision.reason === "low_visibility_direction_withheld") {
                return { text: phrases.LOW_VISIBILITY, priority: SpeechPriority.CAUTION };
            }
            if (decision.reason === "narrow_path") {
                return { text: phrases.NARROW, priority: SpeechPriority.CAUTION };
            }
            if (tier === HazardTier.PERSON) {
                // A person moving across the corridor is worth distinguishing:
                // the right response is to wait, not to swerve.
                const crossing = obstacle && Math.abs(obstacle.velocityX) > 0.12;
                return {
                    text: crossing ? phrases.PERSON_CROSSING : phrases.PERSON_AHEAD,
                    priority: SpeechPriority.CAUTION
                };
            }
            if (tier === HazardTier.VEHICLE) {
                return { text: phrases.VEHICLE_APPROACHING, priority: SpeechPriority.IMMEDIATE_DANGER };
            }
            if (decision.urgency === Urgency.HIGH) {
                return { text: phrases.CAUTION, priority: SpeechPriority.CAUTION };
            }
            return { text: phrases.OBSTACLE_AHEAD, priority: SpeechPriority.CAUTION };
        }

        case Action.CONTINUE:
        default:
            // Silence. "Path clear" is handled separately and only after an
            // obstruction episode.
            return null;
    }
}

export class SpeechPolicy {
    constructor(cfg = SpeechConfig) {
        this.cfg = cfg;
        this.lastSpokenText = "";
        this.lastSpokenAction = null;
        this.lastSpokenSignature = "";
        this.lastSpokenRisk = 0;
        this.lastSpokenAtMs = -Infinity;
        this.lastSpokenPriority = SpeechPriority.SYSTEM;
        this.lastEmergencyAtMs = -Infinity;
        /** Timestamps of recent non-emergency utterances, for the rate backstop. */
        this._recent = [];
        this.counters = {
            spoken: 0,
            suppressedNoChange: 0,
            suppressedCooldown: 0,
            suppressedRateLimit: 0,
            emergencies: 0,
            reminders: 0
        };
        this.lastVerdict = SpeechVerdict.SILENT_CLEAR_PATH;
    }

    /**
     * @param {object} ctx
     * @param {object} ctx.decision
     * @param {number} ctx.nowMs
     * @param {boolean} [ctx.announceClear] decision engine says the path is
     *        confirmed clear after an obstruction
     * @param {boolean} [ctx.enabled]
     * @returns {{speak:boolean, text?:string, priority?:number, emergency?:boolean,
     *            verdict:string, interrupt?:boolean}}
     */
    evaluate({ decision, nowMs, announceClear = false, enabled = true }) {
        if (!enabled) {
            this.lastVerdict = SpeechVerdict.SILENT_DISABLED;
            return { speak: false, verdict: SpeechVerdict.SILENT_DISABLED };
        }

        const emergency = Boolean(decision.emergency)
            || decision.urgency === Urgency.CRITICAL
            || decision.action === Action.STOP;

        const phrase = phraseForDecision(decision, this.cfg.phrases);

        /* ------------------------------------------------- emergency bypass */

        if (emergency && phrase) {
            // Bypasses every cooldown except an anti-stutter floor. If the
            // previous utterance was reassurance, it is cancelled mid-word.
            //
            // The floor is keyed on *priority*, not on the exact text. An earlier
            // version only suppressed identical phrases, which meant a hazard
            // whose distance band wobbled between NEAR and VERY_NEAR alternated
            // between "Stop. Object approaching." and "Stop. Obstacle close." and
            // bypassed the floor every frame. The user needs to hear "stop" once
            // and then be left alone to act on it; which flavour of stop it is
            // does not justify saying it again.
            if (nowMs - this.lastEmergencyAtMs < this.cfg.cooldowns.emergencyMs) {
                this.lastVerdict = SpeechVerdict.SILENT_COOLDOWN;
                this.counters.suppressedCooldown += 1;
                return { speak: false, verdict: SpeechVerdict.SILENT_COOLDOWN };
            }
            this.lastVerdict = SpeechVerdict.EMERGENCY_BYPASS;
            this.counters.emergencies += 1;
            return {
                speak: true,
                text: phrase.text,
                priority: SpeechPriority.STOP,
                emergency: true,
                interrupt: true,
                verdict: SpeechVerdict.EMERGENCY_BYPASS
            };
        }

        /* ------------------------------------------------------- path clear */

        if (decision.action === Action.CONTINUE) {
            if (announceClear) {
                if (nowMs - this.lastSpokenAtMs < this.cfg.cooldowns.changeActionMs) {
                    this.lastVerdict = SpeechVerdict.SILENT_COOLDOWN;
                    this.counters.suppressedCooldown += 1;
                    return { speak: false, verdict: SpeechVerdict.SILENT_COOLDOWN };
                }
                return {
                    speak: true,
                    text: this.cfg.phrases.PATH_CLEAR,
                    priority: SpeechPriority.PATH_STATUS,
                    emergency: false,
                    verdict: SpeechVerdict.SPOKEN
                };
            }
            this.lastVerdict = SpeechVerdict.SILENT_CLEAR_PATH;
            return { speak: false, verdict: SpeechVerdict.SILENT_CLEAR_PATH };
        }

        if (!phrase) {
            this.lastVerdict = SpeechVerdict.SILENT_NO_CHANGE;
            return { speak: false, verdict: SpeechVerdict.SILENT_NO_CHANGE };
        }

        /* -------------------------------------------- material change check */

        const signature = typeof decision.signature === "function" ? decision.signature() : "";
        const risk = decision.primaryObstacle?.risk ?? 0;
        const sameText = phrase.text === this.lastSpokenText;
        const actionChanged = decision.action !== this.lastSpokenAction;
        const signatureChanged = signature !== this.lastSpokenSignature;
        const riskJumped = Math.abs(risk - this.lastSpokenRisk) >= this.cfg.materialRiskDelta;
        const sinceLast = nowMs - this.lastSpokenAtMs;

        let isReminder = false;
        if (!actionChanged && !signatureChanged && !riskJumped) {
            // Nothing changed. Re-speak only as an occasional reminder while the
            // instruction is still live.
            if (sinceLast < this.cfg.reminderMs) {
                this.lastVerdict = SpeechVerdict.SILENT_NO_CHANGE;
                this.counters.suppressedNoChange += 1;
                return { speak: false, verdict: SpeechVerdict.SILENT_NO_CHANGE };
            }
            isReminder = true;
        }

        /* ------------------------------------------------------- cooldowns */

        const cooldown = this._cooldownFor(phrase.priority, sameText);
        if (sinceLast < cooldown) {
            this.lastVerdict = SpeechVerdict.SILENT_COOLDOWN;
            this.counters.suppressedCooldown += 1;
            return { speak: false, verdict: SpeechVerdict.SILENT_COOLDOWN };
        }

        /* ------------------------------------------------- rate backstop */

        this._trimRecent(nowMs);
        if (this._recent.length >= this.cfg.maxUtterancesPerMinute) {
            this.lastVerdict = SpeechVerdict.SILENT_RATE_LIMIT;
            this.counters.suppressedRateLimit += 1;
            return { speak: false, verdict: SpeechVerdict.SILENT_RATE_LIMIT };
        }

        if (isReminder) this.counters.reminders += 1;

        return {
            speak: true,
            text: phrase.text,
            priority: phrase.priority,
            emergency: false,
            reminder: isReminder,
            verdict: SpeechVerdict.SPOKEN
        };
    }

    _cooldownFor(priority, sameText) {
        const c = this.cfg.cooldowns;
        if (priority <= SpeechPriority.IMMEDIATE_DANGER) return c.emergencyMs;
        if (priority === SpeechPriority.DIRECTION) return sameText ? c.repeatSameMs : c.changeActionMs;
        if (priority === SpeechPriority.CAUTION) return sameText ? c.cautionMs : c.changeActionMs;
        if (priority === SpeechPriority.PATH_STATUS) return c.pathStatusMs;
        return c.systemMs;
    }

    _trimRecent(nowMs) {
        const cutoff = nowMs - 60000;
        while (this._recent.length && this._recent[0] < cutoff) this._recent.shift();
    }

    /** Record that an utterance was actually dispatched. */
    commit({ text, priority, emergency, decision, nowMs }) {
        this.lastSpokenText = text;
        this.lastSpokenPriority = priority;
        this.lastSpokenAtMs = nowMs;
        if (emergency) this.lastEmergencyAtMs = nowMs;
        if (decision) {
            this.lastSpokenAction = decision.action;
            this.lastSpokenSignature = typeof decision.signature === "function" ? decision.signature() : "";
            this.lastSpokenRisk = decision.primaryObstacle?.risk ?? 0;
        }
        // Trim here, not only in the rate-limit branch. Emergency and path-status
        // speech reaches `commit` without passing that branch, so trimming there
        // alone let this window grow for the whole session.
        this._trimRecent(nowMs);
        // Emergencies are deliberately not counted against the chatter budget.
        // They have their own floor, and letting them fill the window would make
        // the backstop suppress *directional* guidance after a hazard - starving
        // the user of the instruction they need next.
        if (!emergency) this._recent.push(nowMs);
        this.counters.spoken += 1;
        this.lastVerdict = emergency ? SpeechVerdict.EMERGENCY_BYPASS : SpeechVerdict.SPOKEN;
    }

    metrics() {
        return {
            lastText: this.lastSpokenText,
            lastAction: this.lastSpokenAction,
            lastVerdict: this.lastVerdict,
            utterancesLastMinute: this._recent.length,
            ...this.counters
        };
    }

    reset() {
        this.lastSpokenText = "";
        this.lastSpokenAction = null;
        this.lastSpokenSignature = "";
        this.lastSpokenRisk = 0;
        this.lastSpokenAtMs = -Infinity;
        this.lastEmergencyAtMs = -Infinity;
        this._recent.length = 0;
    }
}

export default SpeechPolicy;
