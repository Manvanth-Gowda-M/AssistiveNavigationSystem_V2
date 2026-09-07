/**
 * Priority queue for speech utterances.
 *
 * Kept from the pre-upgrade build; the pre-emption design was right. Fixed:
 *
 *  - It referenced `SpeechConfig.Priority.DIRECTION_CHANGE`, which no longer
 *    exists after the priority rename. That silently evaluated to `undefined`,
 *    and since every comparison against `undefined` is false, items appended
 *    instead of sorting - the queue looked fine and quietly stopped prioritising.
 *
 *  - Priorities are now validated on the way in, so a bad value is coerced to a
 *    safe default rather than corrupting the ordering.
 *
 * Lower number means higher priority, matching `SpeechPriority`.
 */

import { SpeechPriority } from "../config/speechConfig.js";

/** Pending utterances kept small on purpose: stale guidance is worse than none. */
const MAX_PENDING = 3;

export class SpeechItem {
    constructor({ text, priority = SpeechPriority.DIRECTION, isEmergency = false }) {
        this.text = text;
        this.priority = Number.isFinite(priority) ? priority : SpeechPriority.DIRECTION;
        this.isEmergency = Boolean(isEmergency);
        this.timestamp = Date.now();
    }
}

export class SpeechQueue {
    constructor(maxPending = MAX_PENDING) {
        this.queue = [];
        this.maxPending = maxPending;
        this.dropped = 0;
    }

    /**
     * An emergency replaces the queue outright. Anything already waiting is by
     * definition less important than "Stop.", and speaking it first would delay
     * the only message that matters.
     */
    enqueue(item) {
        if (item.isEmergency || item.priority <= SpeechPriority.STOP) {
            this.dropped += this.queue.length;
            this.queue = [item];
            return;
        }

        let index = this.queue.length;
        for (let i = 0; i < this.queue.length; i += 1) {
            if (item.priority < this.queue[i].priority) {
                index = i;
                break;
            }
        }
        this.queue.splice(index, 0, item);

        // Drop from the tail: the lowest-priority pending items are the ones
        // least worth saying by the time we get to them.
        if (this.queue.length > this.maxPending) {
            this.dropped += this.queue.length - this.maxPending;
            this.queue.length = this.maxPending;
        }
    }

    dequeue() {
        return this.queue.shift() || null;
    }

    clear() {
        this.dropped += this.queue.length;
        this.queue = [];
    }

    peek() {
        return this.queue[0] || null;
    }

    isEmpty() {
        return this.queue.length === 0;
    }

    get length() {
        return this.queue.length;
    }
}

export default SpeechQueue;
