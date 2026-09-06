/**
 * Navigation State Machine Module
 * Manages deterministic system states and transitions.
 */

export const NavState = {
    IDLE: "IDLE",
    INITIALIZING: "INITIALIZING",
    READY: "READY",
    ASSISTING: "ASSISTING",
    CAUTION: "CAUTION",
    OBSTACLE_DETECTED: "OBSTACLE_DETECTED",
    DIRECTION_GUIDANCE: "DIRECTION_GUIDANCE",
    EMERGENCY_STOP: "EMERGENCY_STOP",
    LOW_VISIBILITY: "LOW_VISIBILITY",
    PAUSED: "PAUSED",
    ERROR: "ERROR",
    STOPPED: "STOPPED"
};

export class NavigationStateMachine {
    constructor(onStateChange = null) {
        this.currentState = NavState.IDLE;
        this.previousState = NavState.IDLE;
        this.onStateChange = onStateChange;
    }

    transition(newState, payload = {}) {
        if (this.currentState === newState) return;

        this.previousState = this.currentState;
        this.currentState = newState;
        console.log(`[StateMachine] ${this.previousState} ➔ ${this.currentState}`, payload);

        if (this.onStateChange) {
            this.onStateChange(this.currentState, this.previousState, payload);
        }
    }

    getState() {
        return this.currentState;
    }

    isAssisting() {
        return [
            NavState.ASSISTING,
            NavState.CAUTION,
            NavState.OBSTACLE_DETECTED,
            NavState.DIRECTION_GUIDANCE,
            NavState.EMERGENCY_STOP
        ].includes(this.currentState);
    }
}
