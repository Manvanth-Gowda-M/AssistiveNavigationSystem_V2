/**
 * Frame acquisition and preprocessing.
 *
 * The camera runs at preview resolution for the user's benefit; inference runs
 * on a separate, much smaller letterboxed stream. Those are two different
 * things and conflating them was the biggest single performance defect in the
 * pre-upgrade build.
 *
 * Allocation policy (requirement 12): every canvas, context and typed array
 * here is created once and reused for the life of the session. Tensor buffers
 * are pooled and handed to the worker by transfer, then handed back, so a
 * ten-minute walk allocates the same memory as a ten-second one.
 *
 * The single unavoidable per-inference allocation is the `ImageData` returned
 * by `getImageData` - the 2D canvas API has no read-into-existing-buffer form.
 * It is a fixed 320x320x4 = 410 KB short-lived buffer, it does not grow, and it
 * is the only one. Everything downstream writes into pooled storage.
 */

import { letterboxParams } from "./decoders.js";
import { VisionConfig } from "../config/visionConfig.js";

/**
 * Fixed-size pool of Float32Array tensor buffers.
 *
 * Buffers are transferred to the worker (which detaches them here) and
 * transferred back on completion. `size` of 3 covers one in flight, one being
 * filled, and one spare for a late return.
 */
class TensorBufferPool {
    constructor(elementCount, size = 3) {
        this.elementCount = elementCount;
        this.size = size;
        this._free = [];
        this._outstanding = 0;
        for (let i = 0; i < size; i += 1) {
            this._free.push(new Float32Array(elementCount));
        }
    }

    /** @returns {Float32Array|null} null when everything is in flight. */
    acquire() {
        const buf = this._free.pop();
        if (!buf) return null;
        this._outstanding += 1;
        return buf;
    }

    /** Return a buffer. Accepts an ArrayBuffer (as received back from a worker). */
    release(bufferOrArray) {
        if (!bufferOrArray) {
            this._outstanding = Math.max(0, this._outstanding - 1);
            return;
        }
        const arr = bufferOrArray instanceof Float32Array
            ? bufferOrArray
            : new Float32Array(bufferOrArray);
        if (arr.length !== this.elementCount) {
            // Input size changed mid-session; drop the stale buffer.
            this._outstanding = Math.max(0, this._outstanding - 1);
            return;
        }
        this._outstanding = Math.max(0, this._outstanding - 1);
        if (this._free.length < this.size) this._free.push(arr);
    }

    get available() {
        return this._free.length;
    }

    get outstanding() {
        return this._outstanding;
    }

    stats() {
        return {
            elementCount: this.elementCount,
            poolSize: this.size,
            available: this._free.length,
            outstanding: this._outstanding,
            bytes: this.elementCount * 4 * this.size
        };
    }
}

/**
 * Create a canvas and its 2D context together.
 *
 * `OffscreenCanvas` is preferred because it avoids any layout or compositing
 * involvement. But Safari shipped the constructor before 2D context support, so
 * `typeof OffscreenCanvas !== "undefined"` is not sufficient - on those versions
 * `getContext("2d")` returns null and the pipeline would silently capture nothing.
 * The context is therefore verified here, with a detached `<canvas>` as the
 * fallback.
 *
 * @returns {{canvas:object, ctx:object, kind:string}|null}
 */
function createDrawSurface(width, height) {
    const options = { willReadFrequently: true, alpha: false };

    if (typeof OffscreenCanvas !== "undefined") {
        try {
            const canvas = new OffscreenCanvas(width, height);
            const ctx = canvas.getContext("2d", options);
            if (ctx) return { canvas, ctx, kind: "offscreen" };
        } catch { /* fall through to the element path */ }
    }

    if (typeof document !== "undefined") {
        try {
            const canvas = document.createElement("canvas");
            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext("2d", options);
            if (ctx) return { canvas, ctx, kind: "element" };
        } catch { /* no drawing surface available */ }
    }

    return null;
}

export class FramePipeline {
    /**
     * @param {{inputSize?:number, poolSize?:number}} [opts]
     */
    constructor({ inputSize = VisionConfig.inference.defaultInputSize, poolSize = 3 } = {}) {
        this.inputSize = inputSize;
        this.poolSize = poolSize;
        this.padValue = VisionConfig.inference.padValue;

        /** Inference canvas: exactly the model input size, letterboxed. */
        const inference = createDrawSurface(inputSize, inputSize);
        this._canvas = inference?.canvas || null;
        this._ctx = inference?.ctx || null;
        this.surfaceKind = inference?.kind || "none";

        /** Analysis thumbnail canvas, shared by quality and scene-change stages. */
        const qw = VisionConfig.quality.sampleWidth;
        const qh = VisionConfig.quality.sampleHeight;
        const thumb = createDrawSurface(qw, qh);
        this._thumbCanvas = thumb?.canvas || null;
        this._thumbCtx = thumb?.ctx || null;
        this._thumbWidth = qw;
        this._thumbHeight = qh;
        /** Persistent luminance plane for the analysis stages. */
        this._luma = new Uint8Array(qw * qh);

        this._pool = new TensorBufferPool(inputSize * inputSize * 3, poolSize);
        this._letterbox = letterboxParams(640, 480, inputSize, inputSize);
        this._lastSrcWidth = 0;
        this._lastSrcHeight = 0;
        this._padPainted = false;

        this.framesCaptured = 0;
        this.capturesSkippedNoBuffer = 0;
    }

    /** Change the inference input size, rebuilding only what must be rebuilt. */
    setInputSize(inputSize) {
        if (inputSize === this.inputSize) return false;
        this.inputSize = inputSize;
        if (this._canvas) {
            this._canvas.width = inputSize;
            this._canvas.height = inputSize;
        }
        this._pool = new TensorBufferPool(inputSize * inputSize * 3, this.poolSize);
        this._lastSrcWidth = 0;
        this._lastSrcHeight = 0;
        this._padPainted = false;
        return true;
    }

    get ready() {
        return Boolean(this._ctx);
    }

    /** True when a tensor buffer is free, i.e. a capture can proceed. */
    get canCapture() {
        return this._pool.available > 0;
    }

    /** Most recent letterbox transform; needed to map boxes back. */
    get letterbox() {
        return this._letterbox;
    }

    /**
     * Draw the source into the inference canvas and fill a pooled tensor.
     *
     * @param {HTMLVideoElement|ImageBitmap|HTMLCanvasElement} source
     * @returns {{buffer:Float32Array, letterbox:object, srcWidth:number,
     *            srcHeight:number, drawMs:number, convertMs:number}|null}
     */
    capture(source) {
        if (!this._ctx || !source) return null;

        const srcWidth = source.videoWidth || source.width || 0;
        const srcHeight = source.videoHeight || source.height || 0;
        if (!srcWidth || !srcHeight) return null;

        const buffer = this._pool.acquire();
        if (!buffer) {
            // Every buffer is in flight. Skipping is correct: never queue.
            this.capturesSkippedNoBuffer += 1;
            return null;
        }

        const size = this.inputSize;
        if (srcWidth !== this._lastSrcWidth || srcHeight !== this._lastSrcHeight) {
            this._letterbox = letterboxParams(srcWidth, srcHeight, size, size);
            this._lastSrcWidth = srcWidth;
            this._lastSrcHeight = srcHeight;
            this._padPainted = false;
        }
        const lb = this._letterbox;

        const t0 = performance.now();

        // The padding bars only need repainting when the geometry changes; the
        // image draw overwrites the interior every frame anyway.
        if (!this._padPainted) {
            this._ctx.fillStyle = `rgb(${this.padValue},${this.padValue},${this.padValue})`;
            this._ctx.fillRect(0, 0, size, size);
            this._padPainted = true;
        }

        this._ctx.drawImage(
            source,
            0, 0, srcWidth, srcHeight,
            lb.padX, lb.padY, lb.drawW, lb.drawH
        );

        const t1 = performance.now();

        // Sole per-inference allocation; see the module comment.
        const imageData = this._ctx.getImageData(0, 0, size, size);
        const rgba = imageData.data;

        // RGBA uint8 -> planar RGB float32 in [0,1] (NCHW).
        const plane = size * size;
        const gOff = plane;
        const bOff = plane * 2;
        const inv = 1 / 255;
        for (let p = 0, q = 0; p < plane; p += 1, q += 4) {
            buffer[p] = rgba[q] * inv;
            buffer[gOff + p] = rgba[q + 1] * inv;
            buffer[bOff + p] = rgba[q + 2] * inv;
        }

        const t2 = performance.now();
        this.framesCaptured += 1;

        return {
            buffer,
            letterbox: lb,
            srcWidth,
            srcHeight,
            drawMs: t1 - t0,
            convertMs: t2 - t1
        };
    }

    /** Hand a transferred buffer back to the pool. */
    releaseBuffer(bufferOrArray) {
        this._pool.release(bufferOrArray);
    }

    /**
     * Refresh the shared analysis thumbnail and return its luminance plane.
     *
     * One draw + one read, into a persistent Uint8Array. Called at the analysis
     * cadence, not per rendered frame - the pre-upgrade build sampled twice on
     * every `requestAnimationFrame` tick.
     *
     * @returns {{luma:Uint8Array, width:number, height:number}|null}
     */
    sampleLuminance(source) {
        if (!this._thumbCtx || !source) return null;
        const srcWidth = source.videoWidth || source.width || 0;
        const srcHeight = source.videoHeight || source.height || 0;
        if (!srcWidth || !srcHeight) return null;

        try {
            this._thumbCtx.drawImage(source, 0, 0, this._thumbWidth, this._thumbHeight);
            const data = this._thumbCtx.getImageData(0, 0, this._thumbWidth, this._thumbHeight).data;
            const luma = this._luma;
            for (let i = 0, p = 0; p < luma.length; i += 4, p += 1) {
                // ITU-R BT.601 luma, integer-friendly.
                luma[p] = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
            }
            return { luma, width: this._thumbWidth, height: this._thumbHeight };
        } catch {
            // A tainted or not-yet-ready source; treat as "no sample".
            return null;
        }
    }

    stats() {
        return {
            inputSize: this.inputSize,
            surfaceKind: this.surfaceKind,
            framesCaptured: this.framesCaptured,
            capturesSkippedNoBuffer: this.capturesSkippedNoBuffer,
            pool: this._pool.stats(),
            letterbox: {
                scale: Number(this._letterbox.scale.toFixed(4)),
                padX: Math.round(this._letterbox.padX),
                padY: Math.round(this._letterbox.padY)
            }
        };
    }

    dispose() {
        this._ctx = null;
        this._canvas = null;
        this._thumbCtx = null;
        this._thumbCanvas = null;
    }
}

export { TensorBufferPool };
