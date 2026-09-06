/**
 * Audio-First UI Manager
 * Manages accessibility live regions, high-contrast visual status cards,
 * and screen-reader announcements.
 */

export class AudioFirstUIManager {
    constructor() {
        this.liveRegion = document.getElementById("aria-live-announcer");
        this.statusPill = document.getElementById("main-status-pill");
        this.actionHeadline = document.getElementById("main-action-headline");
        this.reasonSubline = document.getElementById("main-reason-subline");
    }

    announce(text) {
        if (this.liveRegion) {
            this.liveRegion.textContent = "";
            setTimeout(() => {
                this.liveRegion.textContent = text;
            }, 50);
        }
    }

    updateAction(decision) {
        if (!this.actionHeadline) return;

        const actionText = decision.action.replace(/_/g, " ");
        this.actionHeadline.textContent = actionText;

        if (this.reasonSubline) {
            this.reasonSubline.textContent = `Urgency: ${decision.urgency} | ${decision.reason.replace(/_/g, " ")}`;
        }

        // Card styling based on action
        if (this.actionHeadline.parentElement) {
            const card = this.actionHeadline.parentElement;
            card.classList.remove("action-stop", "action-move", "action-clear", "action-caution");

            if (decision.action === "STOP") {
                card.classList.add("action-stop");
            } else if (decision.action.startsWith("MOVE")) {
                card.classList.add("action-move");
            } else if (decision.action === "CONTINUE") {
                card.classList.add("action-clear");
            } else {
                card.classList.add("action-caution");
            }
        }
    }

    setStatus(text, type = "ready") {
        if (this.statusPill) {
            this.statusPill.textContent = text;
            this.statusPill.className = `status-pill ${type}`;
        }
    }
}
