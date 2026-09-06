/**
 * Voice Manager Module
 * Manages Web Speech API voices, voice discovery, language filtering, and speech options.
 */

import { SpeechConfig } from "../config/speechConfig.js";

export class VoiceManager {
    constructor() {
        this.voices = [];
        this.selectedVoice = null;
        this.rate = SpeechConfig.defaults.rate;
        this.pitch = SpeechConfig.defaults.pitch;
        this.volume = SpeechConfig.defaults.volume;
        this.init();
    }

    init() {
        if (!('speechSynthesis' in window)) {
            console.warn("[VoiceManager] SpeechSynthesis API unavailable.");
            return;
        }

        this.loadVoices();
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = () => this.loadVoices();
        }
    }

    loadVoices() {
        if (!('speechSynthesis' in window)) return;
        this.voices = window.speechSynthesis.getVoices();
        if (this.voices.length > 0 && !this.selectedVoice) {
            // Find preferred English / natural voice
            this.selectedVoice = this.voices.find(v => v.lang.startsWith("en") && (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Samantha"))) ||
                                 this.voices.find(v => v.lang.startsWith("en")) ||
                                 this.voices[0];
            console.log("[VoiceManager] Selected voice:", this.selectedVoice ? this.selectedVoice.name : "Default");
        }
    }

    getVoices() {
        return this.voices;
    }

    setVoice(voiceUri) {
        const found = this.voices.find(v => v.voiceURI === voiceUri);
        if (found) this.selectedVoice = found;
    }

    setRate(rate) {
        this.rate = Math.max(0.5, Math.min(2.0, parseFloat(rate)));
    }

    setVolume(volume) {
        this.volume = Math.max(0.0, Math.min(1.0, parseFloat(volume)));
    }
}
