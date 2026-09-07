/**
 * Async safety utilities.
 *
 * Startup is a chain of browser APIs, any one of which can hang rather than
 * fail on some device. A promise that never settles is worse than a rejection:
 * it produces a UI stuck on "Initialising" with nothing to act on. Every await
 * on the startup path is therefore bounded.
 *
 * Pure. No DOM.
 */

/** Raised when a bounded operation runs out of time. */
export class TimeoutError extends Error {
    constructor(label, ms) {
        super(`${label} timed out after ${ms} ms`);
        this.name = "TimeoutError";
        this.label = label;
        this.timeoutMs = ms;
    }
}

/**
 * Reject if `promise` has not settled within `ms`.
 *
 * The timer is always cleared, including on the success path, so a bounded call
 * cannot keep the event loop alive.
 *
 * @template T
 * @param {Promise<T>|T} promise
 * @param {number} ms
 * @param {string} label used in the error message and in the startup trace
 * @returns {Promise<T>}
 */
export function withTimeout(promise, ms, label = "operation") {
    if (!(ms > 0)) return Promise.resolve(promise);

    let timer = null;
    const timeout = new Promise((_, reject) => {
        timer = setTimeout(() => reject(new TimeoutError(label, ms)), ms);
    });

    return Promise.race([Promise.resolve(promise), timeout]).finally(() => {
        if (timer !== null) clearTimeout(timer);
    });
}

/**
 * Bound an operation but treat a timeout or failure as a non-event.
 *
 * For optional probes: not knowing whether WebGPU exists is an acceptable
 * outcome, and far better than hanging while we find out.
 *
 * @template T
 * @param {Promise<T>|T} promise
 * @param {number} ms
 * @param {T} fallback
 * @returns {Promise<T>}
 */
export async function softTimeout(promise, ms, fallback) {
    try {
        return await withTimeout(promise, ms, "probe");
    } catch {
        return fallback;
    }
}

/** Resolve after `ms`. */
export function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
