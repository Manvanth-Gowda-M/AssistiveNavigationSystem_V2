/**
 * Accessible controls.
 *
 * The four critical functions - START, PAUSE, RESUME, STOP - are the first four
 * things in the tab order, have 56 px minimum touch targets, and are reachable by
 * single keystrokes. Everything else is secondary and is grouped after them.
 *
 * Keyboard map:
 *   Space   start / stop      (the one key someone will find without looking)
 *   P       pause / resume
 *   M       mute / unmute
 *   D       debug overlay
 *   I       diagnostics panel
 *   B       benchmark screen
 *
 * Shortcuts are suppressed while focus is in a form control so they cannot fire
 * while a tester is using the scenario picker.
 */

const FORM_TAGS = new Set(["INPUT", "SELECT", "TEXTAREA", "BUTTON"]);

export class AssistanceControls {
    /**
     * @param {object} handlers callbacks; every one is optional
     */
    constructor(handlers = {}) {
        this.handlers = handlers;

        this.btnStart = document.getElementById("btn-start-assist");
        this.btnPause = document.getElementById("btn-pause-assist");
        this.btnStop = document.getElementById("btn-stop-assist");
        this.btnMute = document.getElementById("btn-mute-audio");
        this.btnDebug = document.getElementById("btn-toggle-debug");
        this.btnDiagnostics = document.getElementById("btn-toggle-diagnostics");
        this.btnBenchmark = document.getElementById("btn-toggle-benchmark");
        this.btnDemo = document.getElementById("btn-toggle-demo");
        this.btnDescribe = document.getElementById("btn-describe-scene");
        this.btnRetry = document.getElementById("btn-retry-vision");
        this.selectMode = document.getElementById("select-quality-mode");

        this.isAssisting = false;
        this.isPaused = false;

        this._wire();
        this._wireKeyboard();
    }

    _on(element, event, handler) {
        if (element && handler) element.addEventListener(event, handler);
    }

    _wire() {
        const h = this.handlers;

        this._on(this.btnStart, "click", () => h.onStart?.());
        this._on(this.btnStop, "click", () => h.onStop?.());
        this._on(this.btnDescribe, "click", () => h.onDescribeScene?.());
        this._on(this.btnRetry, "click", () => h.onRetry?.());

        this._on(this.btnPause, "click", () => {
            if (this.isPaused) h.onResume?.();
            else h.onPause?.();
        });

        this._on(this.btnMute, "click", () => h.onToggleMute?.());
        this._on(this.btnDebug, "click", () => h.onToggleDebug?.());
        this._on(this.btnDiagnostics, "click", () => h.onToggleDiagnostics?.());
        this._on(this.btnBenchmark, "click", () => h.onToggleBenchmark?.());
        this._on(this.btnDemo, "click", () => h.onToggleDemo?.());

        this._on(this.selectMode, "change", (event) => h.onQualityModeChange?.(event.target.value));
    }

    _wireKeyboard() {
        window.addEventListener("keydown", (event) => {
            if (event.metaKey || event.ctrlKey || event.altKey) return;
            if (FORM_TAGS.has(event.target?.tagName)) return;

            const h = this.handlers;

            switch (event.code) {
                case "Space":
                    event.preventDefault();
                    if (this.isAssisting) h.onStop?.();
                    else h.onStart?.();
                    return;
                default:
                    break;
            }

            switch (event.key.toLowerCase()) {
                case "p":
                    if (!this.isAssisting) return;
                    event.preventDefault();
                    if (this.isPaused) h.onResume?.();
                    else h.onPause?.();
                    break;
                case "m":
                    h.onToggleMute?.();
                    break;
                case "d":
                    h.onToggleDebug?.();
                    break;
                case "i":
                    h.onToggleDiagnostics?.();
                    break;
                case "b":
                    h.onToggleBenchmark?.();
                    break;
                default:
                    break;
            }
        });
    }

    /* ------------------------------------------------------------- state */

    setAssisting(isAssisting) {
        this.isAssisting = isAssisting;
        if (!isAssisting) this.isPaused = false;

        if (this.btnStart) {
            this.btnStart.hidden = isAssisting;
            this.btnStart.setAttribute("aria-disabled", String(isAssisting));
        }
        if (this.btnPause) this.btnPause.hidden = !isAssisting;
        if (this.btnStop) this.btnStop.hidden = !isAssisting;
        this._refreshPauseLabel();
    }

    setPaused(isPaused) {
        this.isPaused = isPaused;
        this._refreshPauseLabel();
    }

    _refreshPauseLabel() {
        if (!this.btnPause) return;
        const label = this.isPaused ? "Resume" : "Pause";
        this.btnPause.textContent = label;
        this.btnPause.setAttribute("aria-label", `${label} assistance`);
        this.btnPause.setAttribute("aria-pressed", String(this.isPaused));
    }

    /** Start is only meaningful once the vision system has passed its gate. */
    setStartEnabled(enabled, reason = null) {
        if (!this.btnStart) return;
        this.btnStart.disabled = !enabled;
        this.btnStart.setAttribute("aria-disabled", String(!enabled));
        if (reason) this.btnStart.setAttribute("title", reason);
        else this.btnStart.removeAttribute("title");
    }

    setMuted(isMuted) {
        if (!this.btnMute) return;
        this.btnMute.textContent = isMuted ? "Unmute" : "Mute";
        this.btnMute.setAttribute("aria-pressed", String(isMuted));
        this.btnMute.setAttribute("aria-label", isMuted ? "Unmute spoken guidance" : "Mute spoken guidance");
    }

    setToggleState(name, active) {
        const button = {
            debug: this.btnDebug,
            diagnostics: this.btnDiagnostics,
            benchmark: this.btnBenchmark,
            demo: this.btnDemo
        }[name];
        if (!button) return;
        button.setAttribute("aria-pressed", String(active));
        button.classList.toggle("btn-active", active);
    }

    setErrorState(hasError) {
        if (this.btnRetry) this.btnRetry.hidden = !hasError;
    }
}

export default AssistanceControls;
