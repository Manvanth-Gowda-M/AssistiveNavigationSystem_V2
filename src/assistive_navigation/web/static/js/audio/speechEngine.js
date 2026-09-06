/**
 * Guidance Speech Engine Module
 * Formats actionable speech phrases, enforces cooldowns/deduplication,
 * and manages priority preemption using Web Speech API.
 */

import { SpeechConfig } from "../config/speechConfig.js";
import { SpeechQueue, SpeechItem } from "./speechQueue.js";
import { VoiceManager } from "./voiceManager.js";

export class GuidanceSpeechEngine {
    constructor() {
        this.queue = new SpeechQueue();
        this.voiceManager = new VoiceManager();
        this.lastSpokenText = "";
        this.lastSpokenTime = 0;
        this.lastSpokenPriority = SpeechConfig.Priority.PATH_STATUS;
        this.isSpeaking = false;
        this.isEnabled = true;
    }

    /**
     * Translates a navigation decision into a short, calm, actionable speech phrase.
     */
    generatePhrase(decision) {
        const { action, urgency, primaryObstacle, reason } = decision;
        const phrases = SpeechConfig.phrases;

        if (action === "STOP") {
            if (reason === "obstacle_very_close") return phrases.STOP_VERY_CLOSE;
            if (reason === "rapid_approach") return phrases.STOP_RAPID_APPROACH;
            if (reason === "both_flanks_blocked") return phrases.STOP_BOTH_BLOCKED;
            return phrases.STOP;
        }

        const label = primaryObstacle ? primaryObstacle.label : "obstacle";

        if (action === "MOVE_RIGHT") {
            return (label === "person") ? "Person ahead. Move right." : "Obstacle ahead. Move right.";
        }
        if (action === "MOVE_SLIGHTLY_RIGHT") {
            return (label === "person") ? "Person ahead. Move slightly right." : "Obstacle ahead. Move slightly right.";
        }
        if (action === "MOVE_LEFT") {
            return (label === "person") ? "Person ahead. Move left." : "Obstacle ahead. Move left.";
        }
        if (action === "MOVE_SLIGHTLY_LEFT") {
            return (label === "person") ? "Person ahead. Move slightly left." : "Obstacle ahead. Move slightly left.";
        }
        if (action === "CAUTION") {
            return phrases.CAUTION_AHEAD;
        }
        if (action === "CONTINUE") {
            // Silence is golden: do not repeatedly speak "Path clear" unless recovering
            return (reason === "recovered_clear") ? phrases.PATH_CLEAR : "";
        }

        return phrases.OBSTACLE_AHEAD;
    }

    /**
     * Speaks a full natural language scene description.
     */
    speakSceneDescription(descriptionText) {
        if (!this.isEnabled || !descriptionText) return;
        // Priority 6: Non-emergency descriptive speech
        this.queue.enqueue(new SpeechItem({
            text: descriptionText,
            priority: SpeechConfig.Priority.OBJECT_DESCRIPTION
        }));
        this.processQueue();
    }

    /**
     * Submits a navigation decision to the speech engine.
     */
    speakDecision(decision) {
        if (!this.isEnabled) return;

        const text = this.generatePhrase(decision);
        const isEmergency = decision.action === "STOP" || decision.urgency === "CRITICAL";
        const priority = isEmergency ? SpeechConfig.Priority.STOP :
                         (decision.action.startsWith("MOVE") ? SpeechConfig.Priority.DIRECTION_CHANGE :
                         (decision.action === "CAUTION" ? SpeechConfig.Priority.CAUTION : SpeechConfig.Priority.PATH_STATUS));

        const now = Date.now();
        const timeSinceLast = now - this.lastSpokenTime;

        // Cooldown Deduplication: Don't repeat same text unless emergency or cooldown elapsed
        if (!isEmergency) {
            const minGap = (text === this.lastSpokenText) ? SpeechConfig.cooldowns.standardInstructionMs : SpeechConfig.cooldowns.directionChangeMinMs;
            if (timeSinceLast < minGap) {
                return;
            }
        }

        // Emergency STOP preemption: immediately cancel active utterance
        if (isEmergency && window.speechSynthesis) {
            window.speechSynthesis.cancel();
            this.isSpeaking = false;
        }

        this.queue.enqueue(new SpeechItem({ text, priority, isEmergency }));
        this.processQueue();
    }

    /**
     * Speaks arbitrary system status message (e.g., "Vision assistance ready.").
     */
    speakSystemMessage(text, priority = SpeechConfig.Priority.HIGH_RISK_OBSTACLE) {
        if (!this.isEnabled) return;
        this.queue.enqueue(new SpeechItem({ text, priority }));
        this.processQueue();
    }

    processQueue() {
        if (this.isSpeaking || this.queue.isEmpty() || !('speechSynthesis' in window)) return;

        const item = this.queue.dequeue();
        if (!item) return;

        this.isSpeaking = true;
        this.lastSpokenText = item.text;
        this.lastSpokenTime = Date.now();
        this.lastSpokenPriority = item.priority;

        const utterance = new SpeechSynthesisUtterance(item.text);
        if (this.voiceManager.selectedVoice) {
            utterance.voice = this.voiceManager.selectedVoice;
        }
        utterance.rate = this.voiceManager.rate;
        utterance.pitch = this.voiceManager.pitch;
        utterance.volume = this.voiceManager.volume;

        utterance.onend = () => {
            this.isSpeaking = false;
            // Short 100ms breath gap between utterances
            setTimeout(() => this.processQueue(), 100);
        };

        utterance.onerror = (e) => {
            console.warn("[GuidanceSpeechEngine] Speech error:", e);
            this.isSpeaking = false;
            this.processQueue();
        };

        window.speechSynthesis.speak(utterance);
    }

    cancelAll() {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
        }
        this.queue.clear();
        this.isSpeaking = false;
    }
}
