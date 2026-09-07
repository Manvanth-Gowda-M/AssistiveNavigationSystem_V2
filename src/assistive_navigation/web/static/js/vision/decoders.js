/**
 * Model output decoders and letterbox geometry.
 *
 * Deliberately free of DOM, WebGL and ONNX Runtime references so it can run in
 * the inference worker *and* be unit-tested under Node. Every hot path reuses
 * preallocated scratch storage - this code runs 10-20 times a second for
 * minutes at a time, so per-frame allocation here shows up as GC pauses in a
 * walking session.
 */

/**
 * Letterbox transform from a source frame into a square model input.
 *
 * We letterbox rather than stretch because stretching changes aspect ratio,
 * which corrupts the bounding-box height cue that the distance estimate depends
 * on.
 *
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} dstW model input width
 * @param {number} dstH model input height
 * @returns {{scale:number, padX:number, padY:number, drawW:number, drawH:number,
 *            srcW:number, srcH:number, dstW:number, dstH:number}}
 */
export function letterboxParams(srcW, srcH, dstW, dstH) {
    const safeSrcW = Math.max(1, srcW || 1);
    const safeSrcH = Math.max(1, srcH || 1);
    const scale = Math.min(dstW / safeSrcW, dstH / safeSrcH);
    const drawW = safeSrcW * scale;
    const drawH = safeSrcH * scale;
    return {
        scale,
        padX: (dstW - drawW) / 2,
        padY: (dstH - drawH) / 2,
        drawW,
        drawH,
        srcW: safeSrcW,
        srcH: safeSrcH,
        dstW,
        dstH
    };
}

/**
 * Map a box from model-input pixel space back to normalised source space.
 * Writes into `out` to avoid allocating. Values are clamped to [0,1] because a
 * detection may legitimately extend past the frame edge.
 *
 * @param {number[]} out length-4 target, receives [x1,y1,x2,y2]
 */
export function modelBoxToNormalisedSource(x1, y1, x2, y2, lb, out) {
    const invScale = 1 / lb.scale;
    const sx1 = (x1 - lb.padX) * invScale;
    const sy1 = (y1 - lb.padY) * invScale;
    const sx2 = (x2 - lb.padX) * invScale;
    const sy2 = (y2 - lb.padY) * invScale;

    out[0] = clamp01(sx1 / lb.srcW);
    out[1] = clamp01(sy1 / lb.srcH);
    out[2] = clamp01(sx2 / lb.srcW);
    out[3] = clamp01(sy2 / lb.srcH);
    return out;
}

function clamp01(v) {
    if (!(v > 0)) return 0;      // also catches NaN
    return v > 1 ? 1 : v;
}

/** IoU on [x1,y1,x2,y2]. */
export function iou(a, b) {
    const ix1 = a[0] > b[0] ? a[0] : b[0];
    const iy1 = a[1] > b[1] ? a[1] : b[1];
    const ix2 = a[2] < b[2] ? a[2] : b[2];
    const iy2 = a[3] < b[3] ? a[3] : b[3];

    const iw = ix2 - ix1;
    const ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;

    const inter = iw * ih;
    const areaA = (a[2] - a[0]) * (a[3] - a[1]);
    const areaB = (b[2] - b[0]) * (b[3] - b[1]);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
}

/**
 * IoU between two boxes held in a flat [x1,y1,x2,y2,...] array.
 *
 * Takes offsets rather than box objects so the hot NMS loop needs no temporary
 * arrays or accessor closures at all.
 */
export function iouFlat(boxes, offsetA, offsetB) {
    const ax1 = boxes[offsetA];
    const ay1 = boxes[offsetA + 1];
    const ax2 = boxes[offsetA + 2];
    const ay2 = boxes[offsetA + 3];
    const bx1 = boxes[offsetB];
    const by1 = boxes[offsetB + 1];
    const bx2 = boxes[offsetB + 2];
    const by2 = boxes[offsetB + 3];

    const ix1 = ax1 > bx1 ? ax1 : bx1;
    const iy1 = ay1 > by1 ? ay1 : by1;
    const ix2 = ax2 < bx2 ? ax2 : bx2;
    const iy2 = ay2 < by2 ? ay2 : by2;

    const iw = ix2 - ix1;
    const ih = iy2 - iy1;
    if (iw <= 0 || ih <= 0) return 0;

    const inter = iw * ih;
    const union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
    return union > 0 ? inter / union : 0;
}

/**
 * Class-aware greedy NMS over candidate indices.
 *
 * Class-aware matters for navigation: a person standing directly in front of a
 * parked car must not have one box suppressed by the other, because the two
 * carry different hazard priorities.
 *
 * Boxes are read straight out of the flat array by offset. An earlier version
 * took an accessor that returned a shared scratch array, which aliased: reading
 * the comparison box overwrote the candidate box, every IoU evaluated to 1, and
 * every detection after the first of each class was silently suppressed. Passing
 * the buffer directly makes that class of bug impossible.
 *
 * @param {Float32Array} boxes flat [x1,y1,x2,y2] per candidate
 * @param {Float32Array|number[]} scores
 * @param {Int32Array|number[]} classIds
 * @param {number[]} order candidate indices sorted by descending score
 * @param {number} iouThreshold
 * @param {number} maxOut
 * @param {number[]} keepOut output array (cleared by this function)
 * @returns {number[]} keepOut
 */
export function greedyNms(boxes, scores, classIds, order, iouThreshold, maxOut, keepOut) {
    keepOut.length = 0;
    for (let oi = 0; oi < order.length && keepOut.length < maxOut; oi += 1) {
        const i = order[oi];
        const offsetI = i * 4;
        let suppressed = false;
        for (let kj = 0; kj < keepOut.length; kj += 1) {
            const j = keepOut[kj];
            if (classIds[i] !== classIds[j]) continue;
            if (iouFlat(boxes, offsetI, j * 4) > iouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) keepOut.push(i);
    }
    return keepOut;
}

/**
 * Reusable decoder. One instance per worker; scratch buffers grow but are never
 * reallocated per frame.
 */
export class DetectionDecoder {
    constructor() {
        /** @type {Float32Array} flat candidate boxes, 4 per candidate */
        this._boxes = new Float32Array(4 * 512);
        this._scores = new Float32Array(512);
        this._classIds = new Int32Array(512);
        this._order = [];
        this._keep = [];
        this._mapped = [0, 0, 0, 0];
    }

    _ensureCapacity(n) {
        if (this._scores.length >= n) return;
        let cap = this._scores.length;
        while (cap < n) cap *= 2;
        this._boxes = new Float32Array(4 * cap);
        this._scores = new Float32Array(cap);
        this._classIds = new Int32Array(cap);
    }

    /**
     * Decode a YOLOv8-style raw head: [1, 4 + numClasses, numAnchors].
     * Rows 0-3 are cx, cy, w, h in model-input pixels; the remaining rows are
     * per-class confidences already in [0,1].
     *
     * @param {Float32Array} data
     * @param {number[]} dims
     * @param {object} opts
     * @param {object} opts.letterbox result of letterboxParams()
     * @param {string[]} opts.classNames
     * @param {number} opts.scoreThreshold
     * @param {number} opts.iouThreshold
     * @param {number} opts.maxDetections
     * @param {number} [opts.minBoxArea] normalised area floor
     * @param {Set<number>|null} [opts.allowedClassIds] decode-time class filter
     * @returns {Array<{label:string, classId:number, confidence:number, box:number[]}>}
     */
    decodeYoloRaw(data, dims, opts) {
        const {
            letterbox: lb,
            classNames,
            scoreThreshold,
            iouThreshold,
            maxDetections,
            minBoxArea = 0,
            allowedClassIds = null
        } = opts;

        // dims is [1, C, A]; tolerate a missing batch dim.
        const channels = dims.length === 3 ? dims[1] : dims[0];
        const anchors = dims.length === 3 ? dims[2] : dims[1];
        const numClasses = channels - 4;
        if (numClasses <= 0 || anchors <= 0) return [];

        this._ensureCapacity(anchors);

        const boxes = this._boxes;
        const scores = this._scores;
        const classIds = this._classIds;
        const order = this._order;
        order.length = 0;

        let count = 0;

        for (let a = 0; a < anchors; a += 1) {
            // Best class for this anchor. Class filtering happens here so that
            // irrelevant classes cost one comparison and nothing else.
            let bestScore = 0;
            let bestClass = -1;
            for (let c = 0; c < numClasses; c += 1) {
                if (allowedClassIds && !allowedClassIds.has(c)) continue;
                const s = data[(4 + c) * anchors + a];
                if (s > bestScore) {
                    bestScore = s;
                    bestClass = c;
                }
            }
            if (bestClass < 0 || bestScore < scoreThreshold) continue;

            const cx = data[a];
            const cy = data[anchors + a];
            const w = data[2 * anchors + a];
            const h = data[3 * anchors + a];
            if (!(w > 0) || !(h > 0)) continue;

            const half = count * 4;
            boxes[half] = cx - w / 2;
            boxes[half + 1] = cy - h / 2;
            boxes[half + 2] = cx + w / 2;
            boxes[half + 3] = cy + h / 2;
            scores[count] = bestScore;
            classIds[count] = bestClass;
            order.push(count);
            count += 1;
        }

        if (count === 0) return [];

        order.sort((i, j) => scores[j] - scores[i]);
        const keep = greedyNms(boxes, scores, classIds, order, iouThreshold, maxDetections, this._keep);

        return this._materialise(keep, boxes, scores, classIds, lb, classNames, minBoxArea);
    }

    /**
     * Decode an end-to-end head: [1, maxDet, 6] = x1, y1, x2, y2, score, classId
     * in model-input pixels, already NMS-free and sorted by descending score.
     *
     * @returns {Array<{label:string, classId:number, confidence:number, box:number[]}>}
     */
    decodeYoloEndToEnd(data, dims, opts) {
        const {
            letterbox: lb,
            classNames,
            scoreThreshold,
            maxDetections,
            minBoxArea = 0,
            allowedClassIds = null
        } = opts;

        const rows = dims.length === 3 ? dims[1] : dims[0];
        const stride = dims.length === 3 ? dims[2] : dims[1];
        if (rows <= 0 || stride < 6) return [];

        const out = [];
        for (let i = 0; i < rows && out.length < maxDetections; i += 1) {
            const o = i * stride;
            const score = data[o + 4];
            // Rows are score-sorted, so the first sub-threshold row ends the scan.
            if (score < scoreThreshold) break;

            const classId = Math.round(data[o + 5]);
            if (allowedClassIds && !allowedClassIds.has(classId)) continue;

            const mapped = modelBoxToNormalisedSource(
                data[o], data[o + 1], data[o + 2], data[o + 3], lb, this._mapped
            );
            const w = mapped[2] - mapped[0];
            const h = mapped[3] - mapped[1];
            if (w <= 0 || h <= 0 || w * h < minBoxArea) continue;

            out.push({
                label: classNames[classId] || "obstacle",
                classId,
                confidence: score,
                box: [mapped[0], mapped[1], mapped[2], mapped[3]]
            });
        }
        return out;
    }

    _materialise(keep, boxes, scores, classIds, lb, classNames, minBoxArea) {
        const out = [];
        for (let k = 0; k < keep.length; k += 1) {
            const i = keep[k];
            const o = i * 4;
            const mapped = modelBoxToNormalisedSource(
                boxes[o], boxes[o + 1], boxes[o + 2], boxes[o + 3], lb, this._mapped
            );
            const w = mapped[2] - mapped[0];
            const h = mapped[3] - mapped[1];
            if (w <= 0 || h <= 0 || w * h < minBoxArea) continue;

            const classId = classIds[i];
            out.push({
                label: classNames[classId] || "obstacle",
                classId,
                confidence: scores[i],
                box: [mapped[0], mapped[1], mapped[2], mapped[3]]
            });
        }
        return out;
    }
}

/**
 * Normalise a runtime that already returns labelled boxes (MediaPipe, TFJS)
 * into the same shape the tensor decoders emit.
 *
 * @param {Array<{label:string, classId?:number, confidence:number, box:number[]}>} raw
 *        boxes already in normalised source coordinates
 */
export function normaliseNativeDetections(raw, { scoreThreshold, minBoxArea = 0, maxDetections = 32 }) {
    const out = [];
    for (const d of raw) {
        if (out.length >= maxDetections) break;
        if (!d || d.confidence < scoreThreshold) continue;
        const [x1, y1, x2, y2] = d.box;
        const bx1 = clamp01(x1);
        const by1 = clamp01(y1);
        const bx2 = clamp01(x2);
        const by2 = clamp01(y2);
        const w = bx2 - bx1;
        const h = by2 - by1;
        if (w <= 0 || h <= 0 || w * h < minBoxArea) continue;
        out.push({
            label: String(d.label || "obstacle").toLowerCase(),
            classId: Number.isInteger(d.classId) ? d.classId : -1,
            confidence: d.confidence,
            box: [bx1, by1, bx2, by2]
        });
    }
    return out;
}
