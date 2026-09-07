/**
 * IndexedDB model asset cache.
 *
 * Model files are megabytes. Re-downloading them every time the user presses
 * Start is both slow and pointless, and on a walking demo it is the difference
 * between "ready in 400 ms" and "ready in 12 seconds". ONNX Runtime's own
 * deployment guidance calls out IndexedDB for exactly this.
 *
 * Works in both window and worker scopes - the inference worker fetches its own
 * weights, so this module must not touch `document`.
 *
 * Entries are keyed by model key and validated against the URL plus the byte
 * length, so swapping a model asset invalidates the cache without a manual
 * bump.
 */

const DB_NAME = "vision-model-cache";
const DB_VERSION = 1;
const STORE = "models";

let dbPromise = null;

function hasIndexedDb() {
    try {
        return typeof indexedDB !== "undefined" && indexedDB !== null;
    } catch {
        return false;
    }
}

function openDb() {
    if (!hasIndexedDb()) return Promise.resolve(null);
    if (dbPromise) return dbPromise;

    dbPromise = new Promise((resolve) => {
        let request;
        try {
            request = indexedDB.open(DB_NAME, DB_VERSION);
        } catch {
            resolve(null);
            return;
        }

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(STORE)) {
                db.createObjectStore(STORE, { keyPath: "key" });
            }
        };
        request.onsuccess = () => resolve(request.result);
        // A private-browsing or storage-denied context is not an error we should
        // propagate; we simply lose caching.
        request.onerror = () => resolve(null);
        request.onblocked = () => resolve(null);
    });

    return dbPromise;
}

function tx(db, mode) {
    const transaction = db.transaction(STORE, mode);
    return transaction.objectStore(STORE);
}

function requestToPromise(request) {
    return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

/**
 * @typedef {object} CachedModel
 * @property {string} key
 * @property {string} url
 * @property {ArrayBuffer} bytes
 * @property {number} byteLength
 * @property {number} storedAt
 * @property {string|null} etag
 */

/** @returns {Promise<CachedModel|null>} */
export async function readCachedModel(key) {
    const db = await openDb();
    if (!db) return null;
    try {
        const record = await requestToPromise(tx(db, "readonly").get(key));
        if (!record || !record.bytes) return null;
        return record;
    } catch {
        return null;
    }
}

/** @returns {Promise<boolean>} success */
export async function writeCachedModel(entry) {
    const db = await openDb();
    if (!db) return false;
    try {
        await requestToPromise(tx(db, "readwrite").put(entry));
        return true;
    } catch (err) {
        // QuotaExceededError is the realistic failure. Not fatal: the model is
        // already in memory, we just will not have it cached next time.
        console.warn("[modelCache] Could not persist model:", err?.name || err);
        return false;
    }
}

export async function deleteCachedModel(key) {
    const db = await openDb();
    if (!db) return false;
    try {
        await requestToPromise(tx(db, "readwrite").delete(key));
        return true;
    } catch {
        return false;
    }
}

/** Metadata for every cached entry, without loading the bytes into memory. */
export async function listCachedModels() {
    const db = await openDb();
    if (!db) return [];
    try {
        const records = await requestToPromise(tx(db, "readonly").getAll());
        return records.map((r) => ({
            key: r.key,
            url: r.url,
            byteLength: r.byteLength,
            storedAt: r.storedAt
        }));
    } catch {
        return [];
    }
}

export async function clearModelCache() {
    const db = await openDb();
    if (!db) return false;
    try {
        await requestToPromise(tx(db, "readwrite").clear());
        return true;
    } catch {
        return false;
    }
}

/** Best-effort storage usage report for the diagnostics dashboard. */
export async function storageEstimate() {
    try {
        const nav = typeof navigator !== "undefined" ? navigator : null;
        if (nav?.storage?.estimate) {
            const { usage, quota } = await nav.storage.estimate();
            return { usage, quota };
        }
    } catch { /* optional */ }
    return null;
}

/**
 * Download a model with progress reporting, streaming into a single
 * preallocated buffer when the server gives us a content length.
 */
async function downloadWithProgress(url, onProgress) {
    const response = await fetch(url, { cache: "force-cache" });
    if (!response.ok) {
        throw new Error(`Model fetch failed: ${response.status} ${response.statusText} (${url})`);
    }

    const etag = response.headers.get("etag");
    const declared = Number(response.headers.get("content-length")) || 0;

    if (!response.body || typeof response.body.getReader !== "function") {
        const bytes = await response.arrayBuffer();
        onProgress?.({ loaded: bytes.byteLength, total: bytes.byteLength, ratio: 1 });
        return { bytes, etag };
    }

    const reader = response.body.getReader();
    const chunks = [];
    let loaded = 0;

    for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;
        onProgress?.({
            loaded,
            total: declared || 0,
            ratio: declared ? Math.min(1, loaded / declared) : 0
        });
    }

    const out = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
    }
    // Release chunk references promptly; these are multi-megabyte buffers.
    chunks.length = 0;

    return { bytes: out.buffer, etag };
}

/**
 * Fetch model bytes, preferring the IndexedDB copy.
 *
 * @param {object} opts
 * @param {string} opts.key registry key
 * @param {string} opts.url absolute URL
 * @param {number} [opts.expectedBytes] used to detect a stale/truncated entry
 * @param {(p:{stage:string, loaded?:number, total?:number, ratio?:number})=>void} [opts.onProgress]
 * @param {boolean} [opts.allowCache=true]
 * @returns {Promise<{bytes:ArrayBuffer, fromCache:boolean, byteLength:number, elapsedMs:number, persisted:boolean}>}
 */
export async function loadModelBytes({ key, url, expectedBytes = 0, onProgress, allowCache = true }) {
    const started = (typeof performance !== "undefined" ? performance.now() : Date.now());

    if (allowCache) {
        onProgress?.({ stage: "cache-lookup" });
        const cached = await readCachedModel(key);
        const urlMatches = cached && cached.url === url;
        // A tolerance of zero is right here: the byte length is exact metadata,
        // and a mismatch means the asset changed under us.
        const sizeMatches = cached && (!expectedBytes || cached.byteLength === expectedBytes);

        if (cached && urlMatches && sizeMatches) {
            onProgress?.({ stage: "cache-hit", loaded: cached.byteLength, total: cached.byteLength, ratio: 1 });
            return {
                bytes: cached.bytes,
                fromCache: true,
                byteLength: cached.byteLength,
                elapsedMs: Math.round((typeof performance !== "undefined" ? performance.now() : Date.now()) - started),
                persisted: true
            };
        }
        if (cached) {
            // Stale entry: drop it rather than leaving dead megabytes behind.
            await deleteCachedModel(key);
        }
    }

    onProgress?.({ stage: "download", loaded: 0, total: expectedBytes, ratio: 0 });
    const { bytes, etag } = await downloadWithProgress(url, (p) => onProgress?.({ stage: "download", ...p }));

    let persisted = false;
    if (allowCache) {
        onProgress?.({ stage: "persist" });
        persisted = await writeCachedModel({
            key,
            url,
            bytes,
            byteLength: bytes.byteLength,
            storedAt: Date.now(),
            etag: etag || null
        });
    }

    return {
        bytes,
        fromCache: false,
        byteLength: bytes.byteLength,
        elapsedMs: Math.round((typeof performance !== "undefined" ? performance.now() : Date.now()) - started),
        persisted
    };
}

/** Cache state summary for the performance dashboard. */
export async function cacheStatusSummary() {
    if (!hasIndexedDb()) return { available: false, entries: [], totalBytes: 0, estimate: null };
    const entries = await listCachedModels();
    const totalBytes = entries.reduce((sum, e) => sum + (e.byteLength || 0), 0);
    return {
        available: true,
        entries,
        totalBytes,
        estimate: await storageEstimate()
    };
}
