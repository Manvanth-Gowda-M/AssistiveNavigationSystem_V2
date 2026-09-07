/**
 * Shared test helpers.
 *
 * The client sources are plain ES modules with no DOM access at import time, so
 * they can be imported directly here. That is a deliberate design constraint:
 * every module on the reasoning path is testable without a browser.
 */

import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));

/** Absolute path to the static client root. */
export const CLIENT_ROOT = path.resolve(here, "../../src/assistive_navigation/web/static");

/** Import a client module by its path relative to the static root. */
export function clientUrl(relative) {
    return new URL(`file://${path.resolve(CLIENT_ROOT, relative).replace(/\\/g, "/")}`).href;
}

/** Normalised box helper. */
export const box = (x1, y1, x2, y2) => [x1, y1, x2, y2];

/**
 * Build a detection in the tracker's input shape.
 * `classId` is irrelevant to the reasoning stages, so it defaults to -1.
 */
export function detection(label, b, confidence = 0.85, visibility = 1) {
    return { label, classId: -1, confidence, box: [...b], visibility };
}

/** Feed the same detections repeatedly so a track passes the confirmation gate. */
export function confirmTrack(tracker, detections, timestampStart = 1000, stepMs = 80, frames = 4) {
    let t = timestampStart;
    let result = null;
    for (let i = 0; i < frames; i += 1) {
        result = tracker.update(detections.map((d) => ({ ...d, box: [...d.box] })), t);
        t += stepMs;
    }
    return { result, endMs: t };
}

/**
 * Minimal SpeechSynthesis stub.
 *
 * Records what would have been spoken and lets a test control when an utterance
 * "finishes", which is how the queue's forward-progress behaviour is exercised.
 */
export class FakeSynth {
    constructor({ autoEnd = true } = {}) {
        this.autoEnd = autoEnd;
        this.spoken = [];
        this.cancels = 0;
        this.paused = false;
        this._pending = [];
    }

    speak(utterance) {
        this.spoken.push(utterance.text);
        if (this.autoEnd) {
            // Synchronous completion keeps tests deterministic; the real API is
            // async but the manager only cares about ordering.
            queueMicrotask(() => utterance.onend?.({}));
        } else {
            this._pending.push(utterance);
        }
    }

    cancel() {
        this.cancels += 1;
        for (const u of this._pending) u.onerror?.({ error: "canceled" });
        this._pending.length = 0;
    }

    resume() {
        this.paused = false;
    }

    finishNext() {
        const u = this._pending.shift();
        u?.onend?.({});
        return Boolean(u);
    }

    get texts() {
        return this.spoken.slice();
    }
}

/** Deterministic clock. */
export class Clock {
    constructor(start = 0) {
        this.t = start;
    }

    now = () => this.t;

    advance(ms) {
        this.t += ms;
        return this.t;
    }
}

/** Build a synthetic luminance plane. `fn(x, y)` returns 0-255. */
export function lumaPlane(width, height, fn) {
    const out = new Uint8Array(width * height);
    for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
            out[y * width + x] = Math.max(0, Math.min(255, Math.round(fn(x, y))));
        }
    }
    return out;
}
