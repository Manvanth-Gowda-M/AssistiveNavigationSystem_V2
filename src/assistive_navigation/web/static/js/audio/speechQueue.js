/**
 * Speech Queue Module
 * Priority queue for speech utterances with emergency preemption and deduplication.
 */

import { SpeechConfig } from "../config/speechConfig.js";

export class SpeechItem {
    constructor({ text, priority = SpeechConfig.Priority.DIRECTION_CHANGE, isEmergency = false }) {
        this.text = text;
        this.priority = priority;
        this.isEmergency = isEmergency;
        this.timestamp = Date.now();
    }
}

export class SpeechQueue {
    constructor() {
        this.queue = [];
    }

    enqueue(item) {
        // Emergency STOP preemption: clear queue and insert at head
        if (item.isEmergency || item.priority === SpeechConfig.Priority.STOP) {
            this.queue = [item];
            return;
        }

        // Insert sorted by priority (lowest number = highest priority)
        let inserted = false;
        for (let i = 0; i < this.queue.length; i++) {
            if (item.priority < this.queue[i].priority) {
                this.queue.splice(i, 0, item);
                inserted = true;
                break;
            }
        }
        if (!inserted) {
            this.queue.push(item);
        }

        // Keep queue compact (max 3 pending messages)
        if (this.queue.length > 3) {
            this.queue = this.queue.slice(0, 3);
        }
    }

    dequeue() {
        return this.queue.shift() || null;
    }

    clear() {
        this.queue = [];
    }

    peek() {
        return this.queue[0] || null;
    }

    isEmpty() {
        return this.queue.length === 0;
    }
}
