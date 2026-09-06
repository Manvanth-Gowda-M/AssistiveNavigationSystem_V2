/**
 * Accessible Touch Controls & Keyboard Navigation Manager
 * Implements >=48px touch targets, full ARIA roles, and tactile keyboard shortcuts.
 */

export class AssistanceControls {
    constructor({ onStart, onPause, onResume, onStop, onToggleMute, onToggleDebug, onDescribeScene, onToggleNarration }) {
        this.onStart = onStart;
        this.onPause = onPause;
        this.onResume = onResume;
        this.onStop = onStop;
        this.onToggleMute = onToggleMute;
        this.onToggleDebug = onToggleDebug;
        this.onDescribeScene = onDescribeScene;
        this.onToggleNarration = onToggleNarration;

        this.btnStart = document.getElementById("btn-start-assist");
        this.btnPause = document.getElementById("btn-pause-assist");
        this.btnStop = document.getElementById("btn-stop-assist");
        this.btnMute = document.getElementById("btn-mute-audio");
        this.btnDebug = document.getElementById("btn-toggle-debug");
        this.btnDescribe = document.getElementById("btn-describe-scene");
        this.btnNarrate = document.getElementById("btn-toggle-narrate");
        this.viewport = document.querySelector(".camera-viewport-wrapper");

        this.initListeners();
        this.initKeyboardShortcuts();
    }

    initListeners() {
        if (this.btnStart) {
            this.btnStart.addEventListener("click", () => this.onStart());
        }
        if (this.btnDescribe) {
            this.btnDescribe.addEventListener("click", () => this.onDescribeScene && this.onDescribeScene());
        }
        if (this.btnNarrate) {
            this.btnNarrate.addEventListener("click", () => {
                const isActive = this.btnNarrate.getAttribute("data-narrate") === "true";
                const newState = !isActive;
                this.btnNarrate.setAttribute("data-narrate", newState ? "true" : "false");
                this.btnNarrate.textContent = newState ? "🗣️ Narration: ON" : "🗣️ Narration: OFF";
                this.btnNarrate.className = newState ? "btn-large btn-warning" : "btn-large btn-secondary";
                this.onToggleNarration && this.onToggleNarration(newState);
            });
        }
        if (this.viewport) {
            // Double-tap on viewport to trigger scene description
            let lastTap = 0;
            this.viewport.addEventListener("click", () => {
                const now = Date.now();
                if (now - lastTap < 400) {
                    this.onDescribeScene && this.onDescribeScene();
                }
                lastTap = now;
            });
        }
        if (this.btnPause) {
            this.btnPause.addEventListener("click", () => {
                const isPaused = this.btnPause.getAttribute("data-paused") === "true";
                if (isPaused) {
                    this.btnPause.setAttribute("data-paused", "false");
                    this.btnPause.textContent = "⏸️ PAUSE";
                    this.onResume();
                } else {
                    this.btnPause.setAttribute("data-paused", "true");
                    this.btnPause.textContent = "▶️ RESUME";
                    this.onPause();
                }
            });
        }
        if (this.btnStop) {
            this.btnStop.addEventListener("click", () => this.onStop());
        }
        if (this.btnMute) {
            this.btnMute.addEventListener("click", () => this.onToggleMute());
        }
        if (this.btnDebug) {
            this.btnDebug.addEventListener("click", () => this.onToggleDebug());
        }
    }

    initKeyboardShortcuts() {
        window.addEventListener("keydown", (e) => {
            // Ignore if user is typing in an input
            if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;

            if (e.code === "Space") {
                e.preventDefault();
                const isRunning = this.btnStop && this.btnStop.style.display !== "none";
                if (isRunning) this.onStop();
                else this.onStart();
            } else if (e.key === "p" || e.key === "P") {
                if (this.btnPause) this.btnPause.click();
            } else if (e.key === "m" || e.key === "M") {
                if (this.btnMute) this.btnMute.click();
            } else if (e.key === "d" || e.key === "D") {
                if (this.btnDebug) this.btnDebug.click();
            }
        });
    }

    setAssistanceActive(isActive) {
        if (this.btnStart) this.btnStart.style.display = isActive ? "none" : "flex";
        if (this.btnPause) this.btnPause.style.display = isActive ? "flex" : "none";
        if (this.btnStop) this.btnStop.style.display = isActive ? "flex" : "none";
    }
}
