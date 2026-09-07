/**
 * Audio-first status view.
 *
 * Audio is the primary output; this is the visual companion. Its job is to make
 * the system's state unambiguous at a glance, because "is this thing actually
 * running?" is a question the user should never have to ask (requirement 58).
 *
 * Accessibility notes:
 *  - Two live regions. Guidance is `assertive` so it interrupts a screen reader
 *    mid-sentence, exactly like the emergency speech path. Status is `polite` so
 *    lifecycle chatter never talks over guidance.
 *  - The action card is presented as a status, not a heading, so a screen reader
 *    announces changes rather than requiring navigation.
 *  - Nothing conveys meaning by colour alone; every state has text.
 */

import { Action } from "../navigation/decisionEngine.js";

/** The complete set of user-visible states. */
export const UiState = Object.freeze({
    IDLE: "IDLE",
    INITIALIZING: "INITIALIZING",
    VISION_READY: "VISION_READY",
    ASSISTANCE_ACTIVE: "ASSISTANCE_ACTIVE",
    PAUSED: "PAUSED",
    LOW_VISIBILITY: "LOW_VISIBILITY",
    RECOVERING: "RECOVERING",
    VISION_ERROR: "VISION_ERROR",
    STOPPED: "STOPPED",
    BENCHMARKING: "BENCHMARKING"
});

const STATE_PRESENTATION = Object.freeze({
    [UiState.IDLE]: { text: "Idle", tone: "neutral" },
    [UiState.INITIALIZING]: { text: "Initialising", tone: "warning" },
    [UiState.VISION_READY]: { text: "Vision ready", tone: "ready" },
    [UiState.ASSISTANCE_ACTIVE]: { text: "Assistance active", tone: "active" },
    [UiState.PAUSED]: { text: "Paused", tone: "warning" },
    [UiState.LOW_VISIBILITY]: { text: "Low visibility", tone: "warning" },
    [UiState.RECOVERING]: { text: "Recovering", tone: "warning" },
    [UiState.VISION_ERROR]: { text: "Vision error", tone: "danger" },
    [UiState.STOPPED]: { text: "Stopped", tone: "neutral" },
    [UiState.BENCHMARKING]: { text: "Benchmarking", tone: "active" }
});

/** Human-facing text for each action. Mirrors the spoken phrasing. */
const ACTION_PRESENTATION = Object.freeze({
    [Action.STOP]: { headline: "STOP", tone: "stop" },
    [Action.MOVE_LEFT]: { headline: "MOVE LEFT", tone: "move" },
    [Action.MOVE_SLIGHTLY_LEFT]: { headline: "SLIGHTLY LEFT", tone: "move" },
    [Action.MOVE_RIGHT]: { headline: "MOVE RIGHT", tone: "move" },
    [Action.MOVE_SLIGHTLY_RIGHT]: { headline: "SLIGHTLY RIGHT", tone: "move" },
    [Action.CAUTION]: { headline: "CAUTION", tone: "caution" },
    [Action.CONTINUE]: { headline: "CLEAR", tone: "clear" }
});

export class StatusView {
    constructor({
        statusPill = document.getElementById("main-status-pill"),
        headline = document.getElementById("main-action-headline"),
        subline = document.getElementById("main-reason-subline"),
        card = document.getElementById("main-action-card"),
        guidanceRegion = document.getElementById("aria-guidance-announcer"),
        statusRegion = document.getElementById("aria-status-announcer"),
        progressBar = document.getElementById("startup-progress"),
        progressLabel = document.getElementById("startup-progress-label")
    } = {}) {
        this.statusPill = statusPill;
        this.headline = headline;
        this.subline = subline;
        this.card = card;
        this.guidanceRegion = guidanceRegion;
        this.statusRegion = statusRegion;
        this.progressBar = progressBar;
        this.progressLabel = progressLabel;

        this.state = UiState.IDLE;
        this._lastHeadline = "";
        this._lastSubline = "";
        this._lastGuidance = "";
    }

    setState(state, detail = null) {
        this.state = state;
        const presentation = STATE_PRESENTATION[state] || STATE_PRESENTATION[UiState.IDLE];
        const text = detail ? `${presentation.text} · ${detail}` : presentation.text;

        if (this.statusPill) {
            this.statusPill.textContent = text;
            this.statusPill.className = `status-pill ${presentation.tone}`;
        }
        this.announceStatus(text);
        return state;
    }

    /**
     * Reflect a navigation decision.
     *
     * The subline carries the honest detail: what was detected, how close, how
     * confident. It deliberately shows confidence rather than hiding it
     * (requirement 50) - this is a prototype and the UI says so.
     */
    updateDecision(decision, context = {}) {
        if (!this.headline) return;

        const presentation = ACTION_PRESENTATION[decision.action] || ACTION_PRESENTATION[Action.CAUTION];
        if (presentation.headline !== this._lastHeadline) {
            this.headline.textContent = presentation.headline;
            this._lastHeadline = presentation.headline;
        }

        const sub = this._buildSubline(decision, context);
        if (sub !== this._lastSubline && this.subline) {
            this.subline.textContent = sub;
            this._lastSubline = sub;
        }

        if (this.card) {
            this.card.className = `action-card action-${presentation.tone}`;
        }
    }

    _buildSubline(decision, context) {
        const parts = [];
        const obstacle = decision.primaryObstacle;

        if (obstacle) {
            const distance = String(obstacle.distanceCategory || "unknown").toLowerCase().replace("_", " ");
            parts.push(`${obstacle.label} · ${distance} · risk ${obstacle.risk}`);
        } else if (decision.action === Action.CONTINUE) {
            parts.push("No obstacle in path");
        }

        if (decision.action !== Action.CONTINUE) {
            parts.push(`confidence ${Math.round(decision.confidence * 100)}%`);
        }

        if (context.pathScores) {
            const p = context.pathScores;
            parts.push(`L${p.left.score} C${p.center.score} R${p.right.score}`);
        }

        if (decision.held) parts.push("holding");
        if (context.degradationLabel && context.degradationLevel > 1) {
            parts.push(context.degradationLabel);
        }

        return parts.join("  •  ") || "Ready";
    }

    /**
     * Mirror spoken guidance to the assertive live region.
     *
     * Screen-reader users get the same information as speech-synthesis users
     * without both talking at once, because the app's own synthesis can be muted
     * independently.
     */
    announceGuidance(text) {
        if (!this.guidanceRegion || !text) return;
        if (text === this._lastGuidance) {
            // Re-announcing identical text requires clearing first; otherwise
            // assistive technology treats it as unchanged and stays silent.
            this.guidanceRegion.textContent = "";
        }
        this._lastGuidance = text;
        this.guidanceRegion.textContent = text;
    }

    announceStatus(text) {
        if (!this.statusRegion || !text) return;
        this.statusRegion.textContent = text;
    }

    /** Startup progress: model download, warm-up, health check. */
    setProgress({ ratio = null, label = null, visible = true } = {}) {
        if (this.progressBar) {
            this.progressBar.style.display = visible ? "block" : "none";
            if (ratio === null) {
                this.progressBar.removeAttribute("value");
                this.progressBar.setAttribute("aria-valuetext", label || "working");
            } else {
                this.progressBar.value = Math.round(ratio * 100);
                this.progressBar.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));
            }
        }
        if (this.progressLabel && label !== null) {
            this.progressLabel.textContent = label;
        }
    }

    hideProgress() {
        this.setProgress({ visible: false, label: "" });
    }
}

export default StatusView;
