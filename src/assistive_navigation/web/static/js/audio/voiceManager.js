/**
 * Voice selection for the Web Speech API.
 *
 * Kept from the pre-upgrade build; the voice discovery logic was sound. Two
 * changes:
 *
 *  - It now reads `SpeechConfig.voice` rather than the old `SpeechConfig.defaults`.
 *    That rename broke this file, its constructor threw, and because it is
 *    constructed inside `SpeechManager` -> `NavigationApp`, the entire app failed
 *    to start with nothing on screen. A config rename should never be able to do
 *    that, hence the second change.
 *
 *  - Every value now has an inline fallback and the whole constructor is
 *    failure-tolerant. Voice selection is a comfort feature; it must never be
 *    able to prevent the vision system from starting.
 */

import { SpeechConfig } from "../config/speechConfig.js";

/** Hard defaults, independent of config, so this class always has usable values. */
const FALLBACK = { rate: 1.1, pitch: 1.0, volume: 1.0, preferredLanguage: "en-US" };

export class VoiceManager {
    constructor(config = SpeechConfig) {
        const voice = config?.voice || FALLBACK;

        this.voices = [];
        this.selectedVoice = null;
        this.rate = Number.isFinite(voice.rate) ? voice.rate : FALLBACK.rate;
        this.pitch = Number.isFinite(voice.pitch) ? voice.pitch : FALLBACK.pitch;
        this.volume = Number.isFinite(voice.volume) ? voice.volume : FALLBACK.volume;
        this.preferredLanguage = voice.preferredLanguage || FALLBACK.preferredLanguage;
        this.available = false;

        try {
            this.init();
        } catch (error) {
            // Never fatal. The app can guide perfectly well with the platform
            // default voice.
            console.warn("[VoiceManager] Voice discovery failed; using defaults.", error);
        }
    }

    get synth() {
        return typeof window !== "undefined" && "speechSynthesis" in window
            ? window.speechSynthesis
            : null;
    }

    init() {
        if (!this.synth) return;
        this.available = true;

        this.loadVoices();
        // Voices load asynchronously on most platforms and are frequently an
        // empty list on the first call.
        if (this.synth.onvoiceschanged !== undefined) {
            this.synth.onvoiceschanged = () => {
                try {
                    this.loadVoices();
                } catch { /* non-fatal */ }
            };
        }
    }

    loadVoices() {
        if (!this.synth) return;
        this.voices = this.synth.getVoices() || [];
        if (this.voices.length === 0 || this.selectedVoice) return;

        const language = this.preferredLanguage.slice(0, 2);
        const preferredNames = ["Natural", "Google", "Samantha", "Neural"];

        this.selectedVoice =
            this.voices.find((v) => v.lang?.startsWith(language) && preferredNames.some((n) => v.name?.includes(n)))
            || this.voices.find((v) => v.lang?.startsWith(language))
            || this.voices[0]
            || null;
    }

    getVoices() {
        return this.voices;
    }

    setVoice(voiceUri) {
        const found = this.voices.find((v) => v.voiceURI === voiceUri);
        if (found) this.selectedVoice = found;
        return Boolean(found);
    }

    setRate(rate) {
        const value = parseFloat(rate);
        if (Number.isFinite(value)) this.rate = Math.max(0.5, Math.min(2.0, value));
        return this.rate;
    }

    setVolume(volume) {
        const value = parseFloat(volume);
        if (Number.isFinite(value)) this.volume = Math.max(0, Math.min(1, value));
        return this.volume;
    }

    describe() {
        return {
            available: this.available,
            voiceCount: this.voices.length,
            selected: this.selectedVoice?.name || "platform default",
            rate: this.rate
        };
    }
}

export default VoiceManager;
