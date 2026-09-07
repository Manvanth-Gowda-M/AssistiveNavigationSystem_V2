/**
 * In-app scenario runner.
 *
 * Replays the scripted scenarios from `testing/scenarios.js` through the real
 * navigation stack on the real device, with no camera involved. Two uses:
 *
 *   - Verifying on a physical phone that the decision and speech behaviour
 *     matches what the headless tests assert. If they disagree, something is
 *     device-specific and worth knowing about.
 *   - Demonstrating the reasoning without needing to physically arrange ten
 *     obstacles.
 *
 * The same `PipelineHarness` the automated tests use runs here, so there is one
 * implementation of the pipeline wiring, not two.
 */

import { SCENARIOS } from "../testing/scenarios.js";
import { PipelineHarness } from "../testing/pipelineHarness.js";

/**
 * Evaluate a scenario's assertions against a completed run.
 *
 * Mirrors the assertions in the automated suite. Returns a list of
 * human-readable check results rather than throwing, because this runs in front
 * of an operator.
 *
 * @returns {{passed:boolean, checks:Array<{label:string, ok:boolean, detail:string}>}}
 */
export function evaluateScenario(scenario, harness, result) {
    const checks = [];
    const expect = scenario.expect || {};
    const finalAction = result.finalAction;
    const utterances = result.utterances;
    const lastFrame = harness.frames[harness.frames.length - 1];

    const add = (label, ok, detail = "") => checks.push({ label, ok, detail });

    if (expect.action) {
        add(`final action is ${expect.action}`, finalAction === expect.action, `got ${finalAction}`);
    }
    if (expect.finalAction) {
        add(`final action is ${expect.finalAction}`, finalAction === expect.finalAction, `got ${finalAction}`);
    }
    if (expect.notAction) {
        add(`final action is not ${expect.notAction}`, finalAction !== expect.notAction, `got ${finalAction}`);
    }
    if (expect.side) {
        add(`steers ${expect.side}`, Boolean(finalAction && finalAction.includes(expect.side)), `got ${finalAction}`);
    }
    if (expect.notSide) {
        add(
            `never steers ${expect.notSide}`,
            !result.actions.some((a) => a.includes(expect.notSide)),
            result.actions.filter((a) => a.includes(expect.notSide)).join(",") || "none"
        );
    }
    if (expect.notSideAlt) {
        add(
            `never steers ${expect.notSideAlt}`,
            !result.actions.some((a) => a.includes(expect.notSideAlt)),
            "—"
        );
    }
    if (expect.speaks) {
        add(
            `speaks "${expect.speaks}"`,
            utterances.some((u) => u.text.startsWith(expect.speaks)),
            utterances.map((u) => u.text).join(" | ") || "silence"
        );
    }
    if (expect.maxUtterances !== undefined) {
        add(
            `at most ${expect.maxUtterances} utterance(s)`,
            utterances.length <= expect.maxUtterances,
            `spoke ${utterances.length}: ${utterances.map((u) => u.text).join(" | ")}`
        );
    }
    if (expect.maxDirectionReversals !== undefined) {
        const reversals = harness.countDirectionReversals();
        add(
            `at most ${expect.maxDirectionReversals} left/right reversal(s)`,
            reversals <= expect.maxDirectionReversals,
            `${reversals} reversals`
        );
    }
    if (expect.pathClearCount !== undefined) {
        const count = utterances.filter((u) => u.text === "Path clear.").length;
        add(`says "Path clear." exactly ${expect.pathClearCount}x`, count === expect.pathClearCount, `${count}`);
    }
    if (expect.maxConfirmedTracks !== undefined) {
        const maxConfirmed = harness.frames.reduce((m, f) => Math.max(m, f.confirmed), 0);
        add(
            `at most ${expect.maxConfirmedTracks} confirmed track(s)`,
            maxConfirmed <= expect.maxConfirmedTracks,
            `${maxConfirmed}`
        );
    }
    if (expect.tracksAtEnd !== undefined) {
        add(
            `${expect.tracksAtEnd} track(s) remain at the end`,
            (lastFrame?.tracks.length ?? 0) === expect.tracksAtEnd,
            `${lastFrame?.tracks.length ?? 0}`
        );
    }

    return { passed: checks.every((c) => c.ok), checks };
}

/** Run one scenario headlessly and return the run plus its verdict. */
export function runScenario(scenario) {
    const harness = new PipelineHarness();
    const frames = scenario.frames;
    const baseContext = scenario.context || {};

    for (let i = 0; i < frames.length; i += 1) {
        const ctx = { ...baseContext };
        // Scenarios can declare the frame at which the camera turns; that frame
        // is fed with `trustTracks: false`, which is what the live pipeline does
        // when the scene-change detector reports a turn.
        if (scenario.turnAtFrame === i) ctx.trustTracks = false;
        harness.step(frames[i], ctx);
    }

    const result = {
        records: harness.frames,
        actions: harness.frames.map((f) => f.decision.action),
        finalAction: harness.frames.length ? harness.frames[harness.frames.length - 1].decision.action : null,
        utterances: harness.utterances
    };

    return { harness, result, verdict: evaluateScenario(scenario, harness, result) };
}

export class ScenarioRunnerUI {
    /**
     * @param {{select:HTMLSelectElement, runButton:HTMLElement, runAllButton:HTMLElement,
     *          log:HTMLElement}} elements
     */
    constructor({ select, runButton, runAllButton, log }) {
        this.select = select;
        this.runButton = runButton;
        this.runAllButton = runAllButton;
        this.log = log;

        if (this.select) {
            this.select.innerHTML = SCENARIOS
                .map((s) => `<option value="${s.id}">${s.name}</option>`)
                .join("");
        }
        this.runButton?.addEventListener("click", () => this.runSelected());
        this.runAllButton?.addEventListener("click", () => this.runAll());
    }

    _write(html) {
        if (this.log) this.log.innerHTML = html;
    }

    _append(html) {
        if (this.log) this.log.insertAdjacentHTML("beforeend", html);
    }

    runSelected() {
        const scenario = SCENARIOS.find((s) => s.id === this.select?.value);
        if (!scenario) return null;

        const { harness, result, verdict } = runScenario(scenario);
        this._write(`<div class="log-entry log-title">${scenario.name}</div>`);
        this._append(`<div class="log-entry log-note">${scenario.purpose}</div>`);

        for (const check of verdict.checks) {
            this._append(
                `<div class="log-entry ${check.ok ? "log-success" : "log-fail"}">`
                + `${check.ok ? "PASS" : "FAIL"} — ${check.label} <span class="log-detail">(${check.detail})</span></div>`
            );
        }

        const summary = [
            `frames ${harness.frames.length}`,
            `final ${result.finalAction}`,
            `switches ${harness.countActionSwitches()}`,
            `reversals ${harness.countDirectionReversals()}`,
            `spoken ${result.utterances.length}`
        ].join(" · ");
        this._append(`<div class="log-entry log-note">${summary}</div>`);

        if (result.utterances.length) {
            this._append(
                `<div class="log-entry log-note">Speech: ${result.utterances.map((u) => `"${u.text}"`).join(" → ")}</div>`
            );
        } else {
            this._append('<div class="log-entry log-note">Speech: silent</div>');
        }

        return { scenario, harness, result, verdict };
    }

    runAll() {
        this._write('<div class="log-entry log-title">Running all scenarios…</div>');
        let passed = 0;
        const failures = [];

        for (const scenario of SCENARIOS) {
            const { verdict } = runScenario(scenario);
            if (verdict.passed) {
                passed += 1;
                this._append(`<div class="log-entry log-success">PASS — ${scenario.name}</div>`);
            } else {
                const failed = verdict.checks.filter((c) => !c.ok);
                failures.push({ scenario: scenario.name, failed });
                this._append(
                    `<div class="log-entry log-fail">FAIL — ${scenario.name}: `
                    + failed.map((c) => `${c.label} (${c.detail})`).join("; ")
                    + "</div>"
                );
            }
        }

        this._append(
            `<div class="log-entry log-title">${passed} / ${SCENARIOS.length} scenarios passed</div>`
        );
        return { passed, total: SCENARIOS.length, failures };
    }
}

export { SCENARIOS };
export default ScenarioRunnerUI;
