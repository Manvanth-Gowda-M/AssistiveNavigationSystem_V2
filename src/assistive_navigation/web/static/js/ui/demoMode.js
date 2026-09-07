/**
 * Demo mode (requirements 44, 45, 74).
 *
 * The failure this exists to prevent: pressing Start and immediately walking,
 * so the first ten seconds of the demonstration are the model downloading and
 * the first inference paying for shader compilation. That looks broken even
 * though nothing is wrong.
 *
 * Demo mode enforces the gate: model ready AND camera ready AND warm-up complete
 * AND health check passed, *then* "Vision assistance ready.", and only then does
 * the Start control become meaningful.
 *
 * It also carries the ten-station test course, with an observer checklist, so a
 * demo run is a repeatable procedure rather than an improvisation.
 */

/**
 * The controlled test course. Each station names the expected system behaviour,
 * which is what the observer marks against - not "did it detect the chair" but
 * "did it produce the right guidance".
 */
export const DEMO_COURSE = Object.freeze([
    {
        id: 1,
        setup: "Clear corridor, nothing within 4 m",
        expected: "Silence. Status shows CLEAR.",
        checks: ["No speech", "Decision CONTINUE"]
    },
    {
        id: 2,
        setup: "Chair centred in the path",
        expected: "\"Obstacle ahead\" then a slight-move instruction once a side scores clear.",
        checks: ["Speaks once, not repeatedly", "Direction points to the open side"]
    },
    {
        id: 3,
        setup: "Chair to the left, path open ahead",
        expected: "Silence or a single caution. No instruction to move right for nothing.",
        checks: ["No unnecessary direction change"]
    },
    {
        id: 4,
        setup: "Chair to the right, path open ahead",
        expected: "Mirror of station 3.",
        checks: ["Symmetric behaviour", "Left/right not swapped"]
    },
    {
        id: 5,
        setup: "Person standing centred, 2-3 m away",
        expected: "\"Person ahead.\" Then a direction if a side is clear.",
        checks: ["Person distinguished from furniture", "Speaks once"]
    },
    {
        id: 6,
        setup: "Person walks across the corridor left to right",
        expected: "\"Person crossing.\" Guidance settles once they leave the corridor.",
        checks: ["Crossing detected", "Track expires when they leave", "No stale warning"]
    },
    {
        id: 7,
        setup: "Two obstacles: one centre-left, one centre-right, gap between",
        expected: "A single stable direction towards the wider gap.",
        checks: ["No left/right oscillation", "Chosen side is genuinely wider"]
    },
    {
        id: 8,
        setup: "Left blocked, centre partly blocked, right clear",
        expected: "Move right (or slightly right). Held stably.",
        checks: ["Never recommends the blocked side"]
    },
    {
        id: 9,
        setup: "Right blocked, centre partly blocked, left clear",
        expected: "Move left (or slightly left). Held stably.",
        checks: ["Never recommends the blocked side"]
    },
    {
        id: 10,
        setup: "Both sides and centre blocked",
        expected: "\"Stop.\" Immediately, interrupting any speech in progress.",
        checks: ["STOP is immediate", "Interrupts prior utterance", "Says \"Path clear\" once when cleared"]
    }
]);

/** Startup gate stages, in order. */
export const StartupStage = Object.freeze({
    DEVICE_PROFILE: "device-profile",
    MODEL: "model",
    CAMERA: "camera",
    WARMUP: "warmup",
    HEALTH_CHECK: "health-check",
    READY: "ready"
});

const STAGE_LABELS = {
    [StartupStage.DEVICE_PROFILE]: "Profiling device",
    [StartupStage.MODEL]: "Preparing vision model",
    [StartupStage.CAMERA]: "Starting camera",
    [StartupStage.WARMUP]: "Warming up",
    [StartupStage.HEALTH_CHECK]: "Checking health",
    [StartupStage.READY]: "Ready"
};

export class DemoModeController {
    /**
     * @param {{container?:HTMLElement, onToggle?:(enabled:boolean)=>void}} [opts]
     */
    constructor({ container = null, onToggle = () => {} } = {}) {
        this.container = container;
        this.onToggle = onToggle;
        this.enabled = false;
        this.stage = null;
        this.gate = {
            [StartupStage.DEVICE_PROFILE]: false,
            [StartupStage.MODEL]: false,
            [StartupStage.CAMERA]: false,
            [StartupStage.WARMUP]: false,
            [StartupStage.HEALTH_CHECK]: false
        };
        this.results = new Map();
        this.startedAtMs = null;

        if (container) this._buildCourse();
    }

    setEnabled(enabled) {
        this.enabled = enabled;
        document.body.classList.toggle("demo-mode", enabled);
        this.onToggle(enabled);
        return enabled;
    }

    /* ------------------------------------------------------------ the gate */

    markStage(stage, ok = true) {
        this.stage = stage;
        if (stage in this.gate) this.gate[stage] = ok;
        return this.gateSatisfied;
    }

    /** Every prerequisite met. Only then may the demo claim to be ready. */
    get gateSatisfied() {
        return Object.values(this.gate).every(Boolean);
    }

    /** Which prerequisites are still outstanding. */
    get pendingStages() {
        return Object.entries(this.gate).filter(([, ok]) => !ok).map(([stage]) => stage);
    }

    stageLabel(stage = this.stage) {
        return STAGE_LABELS[stage] || "Working";
    }

    /**
     * Fractional progress through the startup gate, for the progress bar.
     * @returns {number} 0..1
     */
    get progress() {
        const entries = Object.values(this.gate);
        const done = entries.filter(Boolean).length;
        return entries.length ? done / entries.length : 0;
    }

    reset() {
        for (const key of Object.keys(this.gate)) this.gate[key] = false;
        this.stage = null;
    }

    /* ----------------------------------------------------------- checklist */

    _buildCourse() {
        this.container.innerHTML = "";
        this.container.setAttribute("aria-label", "Demo test course");

        const intro = document.createElement("p");
        intro.className = "course-intro";
        intro.textContent = "Ten stations. Run them in order with an observer. Mark each as it is verified.";
        this.container.appendChild(intro);

        const list = document.createElement("ol");
        list.className = "course-list";

        for (const station of DEMO_COURSE) {
            const item = document.createElement("li");
            item.className = "course-station";

            const header = document.createElement("div");
            header.className = "course-header";

            const title = document.createElement("strong");
            title.textContent = station.setup;
            header.appendChild(title);

            const pass = document.createElement("button");
            pass.className = "btn-chip pass";
            pass.type = "button";
            pass.textContent = "Pass";
            pass.setAttribute("aria-label", `Mark station ${station.id} as passed`);
            pass.addEventListener("click", () => this._mark(station.id, "pass", item));

            const fail = document.createElement("button");
            fail.className = "btn-chip fail";
            fail.type = "button";
            fail.textContent = "Fail";
            fail.setAttribute("aria-label", `Mark station ${station.id} as failed`);
            fail.addEventListener("click", () => this._mark(station.id, "fail", item));

            header.appendChild(pass);
            header.appendChild(fail);
            item.appendChild(header);

            const expected = document.createElement("p");
            expected.className = "course-expected";
            expected.textContent = station.expected;
            item.appendChild(expected);

            const checks = document.createElement("ul");
            checks.className = "course-checks";
            for (const check of station.checks) {
                const li = document.createElement("li");
                li.textContent = check;
                checks.appendChild(li);
            }
            item.appendChild(checks);

            list.appendChild(item);
        }

        this.container.appendChild(list);

        const footer = document.createElement("div");
        footer.className = "course-footer";

        this.summary = document.createElement("output");
        this.summary.className = "course-summary";
        this.summary.textContent = "0 / 10 verified";
        footer.appendChild(this.summary);

        const exportBtn = document.createElement("button");
        exportBtn.type = "button";
        exportBtn.className = "btn-large btn-secondary";
        exportBtn.textContent = "Copy results";
        exportBtn.addEventListener("click", () => this._copyResults());
        footer.appendChild(exportBtn);

        this.container.appendChild(footer);
    }

    _mark(stationId, outcome, element) {
        this.results.set(stationId, { outcome, at: new Date().toISOString() });
        element.classList.remove("marked-pass", "marked-fail");
        element.classList.add(outcome === "pass" ? "marked-pass" : "marked-fail");
        this._refreshSummary();
    }

    _refreshSummary() {
        if (!this.summary) return;
        const passed = Array.from(this.results.values()).filter((r) => r.outcome === "pass").length;
        const failed = Array.from(this.results.values()).filter((r) => r.outcome === "fail").length;
        this.summary.textContent = `${passed} passed, ${failed} failed, ${DEMO_COURSE.length - this.results.size} remaining`;
    }

    exportResults(sessionSnapshot = null) {
        return {
            generatedAt: new Date().toISOString(),
            stations: DEMO_COURSE.map((station) => ({
                id: station.id,
                setup: station.setup,
                expected: station.expected,
                result: this.results.get(station.id)?.outcome || "not-run"
            })),
            session: sessionSnapshot
        };
    }

    async _copyResults() {
        const text = JSON.stringify(this.exportResults(), null, 2);
        try {
            await navigator.clipboard.writeText(text);
            if (this.summary) this.summary.textContent = "Results copied to clipboard.";
        } catch {
            // Clipboard permission is commonly denied; a visible fallback beats
            // a silent failure.
            if (this.summary) this.summary.textContent = "Clipboard blocked — results logged to console.";
            console.log(text);
        }
        setTimeout(() => this._refreshSummary(), 2500);
    }

    setVisible(visible) {
        if (this.container) this.container.style.display = visible ? "block" : "none";
    }
}

export default DemoModeController;
