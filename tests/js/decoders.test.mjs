/**
 * Model output decoding and letterbox geometry.
 *
 * These transforms sit between the model and every downstream decision. A sign
 * error or an off-by-one here shifts every bounding box, which would silently
 * corrupt left/right guidance - so they are pinned precisely.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    DetectionDecoder,
    greedyNms,
    iou,
    letterboxParams,
    modelBoxToNormalisedSource,
    normaliseNativeDetections
} from "../../src/assistive_navigation/web/static/js/vision/decoders.js";
import { COCO_CLASSES } from "../../src/assistive_navigation/web/static/js/config/classCatalog.js";

test("letterboxParams centres a 4:3 source in a square input", () => {
    const lb = letterboxParams(640, 480, 320, 320);
    assert.equal(lb.scale, 0.5);
    assert.equal(lb.drawW, 320);
    assert.equal(lb.drawH, 240);
    assert.equal(lb.padX, 0);
    assert.equal(lb.padY, 40);
});

test("letterboxParams centres a portrait source", () => {
    const lb = letterboxParams(480, 640, 320, 320);
    assert.equal(lb.scale, 0.5);
    assert.equal(lb.padX, 40);
    assert.equal(lb.padY, 0);
});

test("modelBoxToNormalisedSource inverts the letterbox exactly", () => {
    const lb = letterboxParams(640, 480, 320, 320);
    const out = [0, 0, 0, 0];

    // A box covering the whole drawn region must map back to the whole frame.
    modelBoxToNormalisedSource(0, 40, 320, 280, lb, out);
    assert.deepEqual(out.map((v) => Number(v.toFixed(6))), [0, 0, 1, 1]);

    // A centred quarter-size box.
    modelBoxToNormalisedSource(80, 100, 240, 220, lb, out);
    assert.deepEqual(out.map((v) => Number(v.toFixed(4))), [0.25, 0.25, 0.75, 0.75]);
});

test("modelBoxToNormalisedSource clamps content that falls in the padding", () => {
    const lb = letterboxParams(640, 480, 320, 320);
    const out = [0, 0, 0, 0];
    // y1 = 10 sits inside the top padding bar, so it clamps to 0 rather than
    // producing a negative coordinate.
    modelBoxToNormalisedSource(-20, 10, 340, 300, lb, out);
    assert.equal(out[0], 0);
    assert.equal(out[1], 0);
    assert.equal(out[2], 1);
    assert.equal(out[3], 1);
});

test("iou is symmetric, 1 for identity and 0 for disjoint boxes", () => {
    const a = [0.1, 0.1, 0.5, 0.5];
    const b = [0.3, 0.3, 0.7, 0.7];
    assert.equal(iou(a, a), 1);
    assert.equal(iou(a, [0.6, 0.6, 0.9, 0.9]), 0);
    assert.equal(iou(a, b), iou(b, a));
    // Intersection 0.2x0.2 = 0.04; union = 0.16 + 0.16 - 0.04 = 0.28
    assert.equal(Number(iou(a, b).toFixed(6)), Number((0.04 / 0.28).toFixed(6)));
});

test("greedyNms suppresses overlapping boxes of the same class only", () => {
    const boxes = Float32Array.from([
        0.1, 0.1, 0.5, 0.5,
        0.12, 0.12, 0.52, 0.52,
        0.11, 0.11, 0.51, 0.51
    ]);
    const scores = [0.9, 0.8, 0.7];

    const sameClass = greedyNms(boxes, scores, [0, 0, 0], [0, 1, 2], 0.45, 10, []);
    assert.deepEqual(sameClass, [0], "heavily overlapping same-class boxes collapse to one");

    // A person standing in front of a car must not have either box suppressed.
    const differentClasses = greedyNms(boxes, scores, [0, 2, 5], [0, 1, 2], 0.45, 10, []);
    assert.deepEqual(differentClasses, [0, 1, 2]);
});

test("greedyNms keeps distinct same-class boxes that do not overlap enough", () => {
    // Regression guard. An earlier implementation read candidate boxes through a
    // shared scratch array, so comparing box i against box j overwrote box i:
    // every IoU came out as 1 and every detection after the first of a class was
    // silently dropped. Two people standing apart is the everyday case that broke.
    const boxes = Float32Array.from([
        0.05, 0.30, 0.25, 0.90,   // person on the left
        0.60, 0.30, 0.80, 0.90,   // person on the right, no overlap at all
        0.30, 0.35, 0.50, 0.88    // person in the middle
    ]);
    const scores = [0.9, 0.85, 0.8];

    const keep = greedyNms(boxes, scores, [0, 0, 0], [0, 1, 2], 0.45, 10, []);
    assert.deepEqual(keep, [0, 1, 2], "three separate people must all survive NMS");
});

test("greedyNms respects the output cap", () => {
    const boxes = Float32Array.from([
        0.00, 0.0, 0.10, 1.0,
        0.20, 0.0, 0.30, 1.0,
        0.40, 0.0, 0.50, 1.0,
        0.60, 0.0, 0.70, 1.0
    ]);
    const keep = greedyNms(boxes, [0.9, 0.8, 0.7, 0.6], [0, 0, 0, 0], [0, 1, 2, 3], 0.45, 2, []);
    assert.equal(keep.length, 2);
});

/** Build a YOLO raw tensor [1, 4+numClasses, anchors] with one planted box. */
function buildRawTensor({ anchors, numClasses, planted }) {
    const data = new Float32Array((4 + numClasses) * anchors);
    for (const p of planted) {
        data[p.anchor] = p.cx;
        data[anchors + p.anchor] = p.cy;
        data[2 * anchors + p.anchor] = p.w;
        data[3 * anchors + p.anchor] = p.h;
        data[(4 + p.classId) * anchors + p.anchor] = p.score;
    }
    return data;
}

test("decodeYoloRaw recovers a planted box in source coordinates", () => {
    const anchors = 16;
    const numClasses = 80;
    const lb = letterboxParams(640, 480, 320, 320);
    const data = buildRawTensor({
        anchors,
        numClasses,
        planted: [{ anchor: 5, cx: 160, cy: 160, w: 64, h: 48, classId: 0, score: 0.9 }]
    });

    const decoder = new DetectionDecoder();
    const results = decoder.decodeYoloRaw(data, [1, 4 + numClasses, anchors], {
        letterbox: lb,
        classNames: COCO_CLASSES,
        scoreThreshold: 0.3,
        iouThreshold: 0.45,
        maxDetections: 10
    });

    assert.equal(results.length, 1);
    assert.equal(results[0].label, "person");
    assert.equal(Number(results[0].confidence.toFixed(4)), 0.9);
    // Model box 128..192 x 136..184 -> source 256..384 x 192..288 -> normalised.
    assert.deepEqual(results[0].box.map((v) => Number(v.toFixed(4))), [0.4, 0.4, 0.6, 0.6]);
});

test("decodeYoloRaw honours the score threshold and the class filter", () => {
    const anchors = 8;
    const numClasses = 80;
    const lb = letterboxParams(320, 320, 320, 320);
    const decoder = new DetectionDecoder();

    const data = buildRawTensor({
        anchors,
        numClasses,
        planted: [
            { anchor: 0, cx: 100, cy: 100, w: 40, h: 40, classId: 0, score: 0.9 },   // person
            { anchor: 1, cx: 200, cy: 200, w: 40, h: 40, classId: 63, score: 0.85 }, // laptop
            { anchor: 2, cx: 260, cy: 100, w: 40, h: 40, classId: 2, score: 0.2 }    // car, low
        ]
    });

    const base = {
        letterbox: lb,
        classNames: COCO_CLASSES,
        scoreThreshold: 0.3,
        iouThreshold: 0.45,
        maxDetections: 10
    };

    const unfiltered = decoder.decodeYoloRaw(data, [1, 84, anchors], base);
    assert.deepEqual(unfiltered.map((d) => d.label).sort(), ["laptop", "person"]);

    // Class filtering happens during decode, so an excluded class costs nothing
    // downstream.
    const filtered = decoder.decodeYoloRaw(data, [1, 84, anchors], {
        ...base,
        allowedClassIds: new Set([0, 2])
    });
    assert.deepEqual(filtered.map((d) => d.label), ["person"]);
});

test("decodeYoloRaw drops sub-minimum-area boxes", () => {
    const anchors = 4;
    const lb = letterboxParams(320, 320, 320, 320);
    const decoder = new DetectionDecoder();
    const data = buildRawTensor({
        anchors,
        numClasses: 80,
        // 4x4 px in a 320x320 input is 0.000156 of the frame: noise.
        planted: [{ anchor: 0, cx: 160, cy: 160, w: 4, h: 4, classId: 0, score: 0.95 }]
    });

    const results = decoder.decodeYoloRaw(data, [1, 84, anchors], {
        letterbox: lb,
        classNames: COCO_CLASSES,
        scoreThreshold: 0.3,
        iouThreshold: 0.45,
        maxDetections: 10,
        minBoxArea: 0.0015
    });
    assert.equal(results.length, 0);
});

test("decodeYoloEndToEnd reads xyxy rows and stops at the score threshold", () => {
    const rows = 4;
    const stride = 6;
    const data = new Float32Array(rows * stride);
    // Row 0: high score person. Row 1: mid score car. Row 2: below threshold.
    data.set([64, 64, 128, 192, 0.91, 0], 0);
    data.set([160, 96, 256, 224, 0.55, 2], stride);
    data.set([10, 10, 40, 40, 0.10, 0], stride * 2);
    data.set([20, 20, 50, 50, 0.05, 0], stride * 3);

    const decoder = new DetectionDecoder();
    const lb = letterboxParams(320, 320, 320, 320);
    const results = decoder.decodeYoloEndToEnd(data, [1, rows, stride], {
        letterbox: lb,
        classNames: COCO_CLASSES,
        scoreThreshold: 0.3,
        maxDetections: 10
    });

    assert.equal(results.length, 2, "scan stops at the first sub-threshold row");
    assert.equal(results[0].label, "person");
    assert.equal(results[1].label, "car");
    assert.deepEqual(results[0].box.map((v) => Number(v.toFixed(4))), [0.2, 0.2, 0.4, 0.6]);
});

test("normaliseNativeDetections clamps, filters and caps", () => {
    const out = normaliseNativeDetections(
        [
            { label: "Person", confidence: 0.9, box: [-0.1, 0.2, 0.5, 0.9] },
            { label: "chair", confidence: 0.1, box: [0.1, 0.1, 0.2, 0.2] },
            { label: "car", confidence: 0.8, box: [0.5, 0.5, 0.5, 0.9] }
        ],
        { scoreThreshold: 0.3, minBoxArea: 0.001, maxDetections: 5 }
    );

    assert.equal(out.length, 1, "low score and zero-width boxes are dropped");
    assert.equal(out[0].label, "person");
    assert.equal(out[0].box[0], 0, "negative coordinates clamp to 0");
});

test("the decoder reuses its scratch buffers across calls", () => {
    const decoder = new DetectionDecoder();
    const before = decoder._scores;
    const anchors = 32;
    const lb = letterboxParams(320, 320, 320, 320);
    const data = buildRawTensor({
        anchors,
        numClasses: 80,
        planted: [{ anchor: 3, cx: 100, cy: 100, w: 60, h: 60, classId: 0, score: 0.8 }]
    });
    const opts = {
        letterbox: lb,
        classNames: COCO_CLASSES,
        scoreThreshold: 0.3,
        iouThreshold: 0.45,
        maxDetections: 10
    };

    for (let i = 0; i < 25; i += 1) decoder.decodeYoloRaw(data, [1, 84, anchors], opts);
    assert.equal(decoder._scores, before, "no reallocation while capacity is sufficient");
});
