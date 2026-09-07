/**
 * Decoder parity against a second implementation, on real model output.
 *
 * The hand-planted tensors in `decoders.test.mjs` prove the decoder does what it
 * was written to do. This file proves it does the *right* thing: the fixtures are
 * genuine ONNX output from both exported detectors, and the expected boxes come
 * from an independent NumPy postprocess (`tools/make_decode_fixtures.py`).
 *
 * If the browser decoder and the reference disagree on the same bytes, one of
 * them is wrong - and a silent disagreement here would shift every bounding box
 * in the live system.
 *
 * Regenerate the fixtures after re-exporting the models:
 *     python tools/make_decode_fixtures.py
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
    DetectionDecoder,
    letterboxParams
} from "../../src/assistive_navigation/web/static/js/vision/decoders.js";
import { COCO_CLASSES } from "../../src/assistive_navigation/web/static/js/config/classCatalog.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE_PATH = path.join(here, "fixtures", "decodeFixtures.json");

const available = fs.existsSync(FIXTURE_PATH);
const fixtures = available ? JSON.parse(fs.readFileSync(FIXTURE_PATH, "utf8")) : null;

/** Tolerance: 1e-4 of frame width is a fraction of a pixel at any sane resolution. */
const BOX_TOLERANCE = 1e-4;
const SCORE_TOLERANCE = 1e-5;

function tensorFrom(model) {
    const bytes = Buffer.from(model.dataBase64, "base64");
    // Copy into an aligned buffer: Buffer slices are not guaranteed to sit on a
    // 4-byte boundary, and Float32Array requires one.
    const aligned = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(aligned).set(bytes);
    return new Float32Array(aligned);
}

test("decode fixtures are present", { skip: available ? false : "run tools/make_decode_fixtures.py" }, () => {
    assert.ok(Object.keys(fixtures.models).length >= 1);
});

for (const [name, model] of Object.entries(fixtures?.models || {})) {
    test(`${name}: browser decoder matches the reference implementation`, () => {
        const data = tensorFrom(model);
        const expectedElements = model.dims.reduce((a, b) => a * b, 1);
        assert.equal(data.length, expectedElements, "fixture tensor length does not match its dims");

        const lb = letterboxParams(
            fixtures.srcWidth,
            fixtures.srcHeight,
            fixtures.inputSize,
            fixtures.inputSize
        );
        const decoder = new DetectionDecoder();
        const options = {
            letterbox: lb,
            classNames: COCO_CLASSES,
            scoreThreshold: fixtures.scoreThreshold,
            iouThreshold: fixtures.iouThreshold,
            maxDetections: fixtures.maxDetections,
            // The reference applies no area floor or class filter, so neither
            // does this comparison.
            minBoxArea: 0
        };

        const actual = model.family === "raw"
            ? decoder.decodeYoloRaw(data, model.dims, options)
            : decoder.decodeYoloEndToEnd(data, model.dims, options);

        assert.equal(
            actual.length,
            model.expected.length,
            `detection count differs: browser ${actual.length}, reference ${model.expected.length}`
        );

        for (let i = 0; i < model.expected.length; i += 1) {
            const want = model.expected[i];
            const got = actual[i];

            assert.equal(got.classId, want.classId, `detection ${i}: class differs`);
            assert.equal(
                got.label,
                COCO_CLASSES[want.classId],
                `detection ${i}: label does not match the class id`
            );
            assert.ok(
                Math.abs(got.confidence - want.confidence) < SCORE_TOLERANCE,
                `detection ${i}: confidence ${got.confidence} vs ${want.confidence}`
            );
            for (let k = 0; k < 4; k += 1) {
                assert.ok(
                    Math.abs(got.box[k] - want.box[k]) < BOX_TOLERANCE,
                    `detection ${i}: box[${k}] ${got.box[k]} vs ${want.box[k]}`
                );
            }
        }
    });

    test(`${name}: raising the threshold monotonically reduces detections`, () => {
        const data = tensorFrom(model);
        const lb = letterboxParams(
            fixtures.srcWidth,
            fixtures.srcHeight,
            fixtures.inputSize,
            fixtures.inputSize
        );
        const decoder = new DetectionDecoder();

        let previous = Infinity;
        for (const threshold of [fixtures.scoreThreshold, 0.01, 0.1, 0.3, 0.6]) {
            const results = model.family === "raw"
                ? decoder.decodeYoloRaw(data, model.dims, {
                    letterbox: lb,
                    classNames: COCO_CLASSES,
                    scoreThreshold: threshold,
                    iouThreshold: fixtures.iouThreshold,
                    maxDetections: fixtures.maxDetections,
                    minBoxArea: 0
                })
                : decoder.decodeYoloEndToEnd(data, model.dims, {
                    letterbox: lb,
                    classNames: COCO_CLASSES,
                    scoreThreshold: threshold,
                    maxDetections: fixtures.maxDetections,
                    minBoxArea: 0
                });

            assert.ok(results.length <= previous, `count rose at threshold ${threshold}`);
            for (const detection of results) {
                assert.ok(detection.confidence >= threshold, "a sub-threshold detection was returned");
            }
            previous = results.length;
        }
    });

    test(`${name}: every decoded box is inside the frame`, () => {
        const data = tensorFrom(model);
        const lb = letterboxParams(
            fixtures.srcWidth,
            fixtures.srcHeight,
            fixtures.inputSize,
            fixtures.inputSize
        );
        const decoder = new DetectionDecoder();
        const results = model.family === "raw"
            ? decoder.decodeYoloRaw(data, model.dims, {
                letterbox: lb,
                classNames: COCO_CLASSES,
                scoreThreshold: fixtures.scoreThreshold,
                iouThreshold: fixtures.iouThreshold,
                maxDetections: fixtures.maxDetections,
                minBoxArea: 0
            })
            : decoder.decodeYoloEndToEnd(data, model.dims, {
                letterbox: lb,
                classNames: COCO_CLASSES,
                scoreThreshold: fixtures.scoreThreshold,
                maxDetections: fixtures.maxDetections,
                minBoxArea: 0
            });

        assert.ok(results.length > 0, "the fixture should yield some detections to check");
        for (const detection of results) {
            const [x1, y1, x2, y2] = detection.box;
            for (const v of detection.box) {
                assert.ok(v >= 0 && v <= 1, `coordinate ${v} is outside the frame`);
            }
            assert.ok(x2 >= x1 && y2 >= y1, "box edges are inverted");
        }
    });
}
