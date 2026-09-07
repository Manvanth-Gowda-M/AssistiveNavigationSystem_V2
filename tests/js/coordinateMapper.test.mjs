/**
 * Coordinate space conversions.
 *
 * "Actual left must equal reported left" is a safety property, not a cosmetic
 * one. These tests pin the crop, rotation and mirroring behaviour that property
 * depends on.
 */

import test from "node:test";
import assert from "node:assert/strict";

import {
    CoordinateMapper,
    ObjectFit,
    computeFitTransform,
    mirrorNormalisedBox,
    rotateNormalisedBox
} from "../../src/assistive_navigation/web/static/js/vision/coordinateMapper.js";
import { box } from "./helpers.mjs";

test("cover crops the long axis and reports the visible region", () => {
    // 4:3 sensor frame shown in a portrait 9:16 box: the sides get cropped.
    const fit = computeFitTransform(640, 480, 360, 640, ObjectFit.COVER);
    assert.equal(fit.scaleX, fit.scaleY);
    assert.equal(fit.scaleX, 640 / 480, "scales to fill the taller axis");

    // Rendered width is 480 * (640/480) ... i.e. 853.33 for a 360 wide box.
    assert.ok(fit.offsetX < 0, "horizontal overflow means cropping");
    assert.equal(fit.offsetY, 0);
    assert.ok(fit.visible.x1 > 0 && fit.visible.x2 < 1, "left and right are cropped away");
    assert.equal(fit.visible.y1, 0);
    assert.equal(fit.visible.y2, 1);
});

test("contain letterboxes and keeps everything visible", () => {
    const fit = computeFitTransform(640, 480, 360, 640, ObjectFit.CONTAIN);
    assert.ok(fit.offsetY > 0, "vertical bars");
    assert.deepEqual(fit.visible, { x1: 0, y1: 0, x2: 1, y2: 1 });
});

test("fill stretches without cropping", () => {
    const fit = computeFitTransform(640, 480, 360, 640, ObjectFit.FILL);
    assert.notEqual(fit.scaleX, fit.scaleY);
    assert.deepEqual(fit.visible, { x1: 0, y1: 0, x2: 1, y2: 1 });
});

test("rotateNormalisedBox handles all four quadrants", () => {
    const b = box(0.1, 0.2, 0.3, 0.5);
    assert.deepEqual(rotateNormalisedBox(b, 0).map((v) => Number(v.toFixed(4))), [0.1, 0.2, 0.3, 0.5]);
    // 90 degrees clockwise: (x,y) -> (1-y, x)
    assert.deepEqual(rotateNormalisedBox(b, 90).map((v) => Number(v.toFixed(4))), [0.5, 0.1, 0.8, 0.3]);
    assert.deepEqual(rotateNormalisedBox(b, 180).map((v) => Number(v.toFixed(4))), [0.7, 0.5, 0.9, 0.8]);
    assert.deepEqual(rotateNormalisedBox(b, 270).map((v) => Number(v.toFixed(4))), [0.2, 0.7, 0.5, 0.9]);
});

test("rotating four times returns the original box", () => {
    let b = box(0.15, 0.25, 0.45, 0.65);
    for (let i = 0; i < 4; i += 1) b = rotateNormalisedBox(b, 90, [0, 0, 0, 0]);
    assert.deepEqual(b.map((v) => Number(v.toFixed(6))), [0.15, 0.25, 0.45, 0.65]);
});

test("mirroring swaps and reflects the horizontal edges", () => {
    assert.deepEqual(mirrorNormalisedBox(box(0.1, 0.2, 0.4, 0.6)), [0.6, 0.2, 0.9, 0.6]);
});

test("navigation space rescales into the visible region", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(360, 640, ObjectFit.COVER);

    const visible = mapper.visibleCameraRegion;
    // A box occupying the exact visible strip must span all of navigation space.
    const full = mapper.cameraToNavigation(box(visible.x1, 0, visible.x2, 1));
    assert.deepEqual(full.box.map((v) => Number(v.toFixed(4))), [0, 0, 1, 1]);
    assert.equal(full.visibility, 1);
});

test("an object cropped out of view is reported as outside", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(360, 640, ObjectFit.COVER);

    const visible = mapper.visibleCameraRegion;
    const outside = mapper.cameraToNavigation(box(0, 0.4, visible.x1 * 0.5, 0.8));
    assert.equal(outside.fullyOutside, true);
    assert.equal(outside.visibility, 0);
});

test("a half-cropped object reports partial visibility", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(360, 640, ObjectFit.COVER);

    const visible = mapper.visibleCameraRegion;
    const width = 0.1;
    // Straddle the left crop edge exactly in half.
    const straddling = mapper.cameraToNavigation(
        box(visible.x1 - width / 2, 0.5, visible.x1 + width / 2, 0.9)
    );
    assert.ok(straddling.visibility > 0.4 && straddling.visibility < 0.6, `got ${straddling.visibility}`);
});

test("mirroring flips navigation left and right", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(640, 480, ObjectFit.CONTAIN);

    const leftObject = box(0.05, 0.5, 0.25, 0.9);
    const unmirrored = mapper.cameraToNavigation(leftObject).box;
    assert.ok(unmirrored[2] < 0.5, "unmirrored: object stays on the left");

    mapper.setOrientation({ mirrored: true, facingUser: true });
    const mirrored = mapper.cameraToNavigation(leftObject).box;
    assert.ok(mirrored[0] > 0.5, "mirrored: object moves to the right");
});

test("display mapping and inverse round-trip", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(360, 640, ObjectFit.COVER);

    const cameraPoint = { x: 0.5, y: 0.5 };
    const displayed = mapper.cameraToDisplay(box(cameraPoint.x, cameraPoint.y, cameraPoint.x, cameraPoint.y));
    const back = mapper.displayToCamera(displayed[0], displayed[1]);
    assert.equal(Number(back.x.toFixed(5)), 0.5);
    assert.equal(Number(back.y.toFixed(5)), 0.5);
});

test("geometry setters report whether anything changed", () => {
    const mapper = new CoordinateMapper();
    // The constructor already seeds 640x480, so use different dimensions here.
    assert.equal(mapper.setVideoGeometry(1280, 720), true);
    assert.equal(mapper.setVideoGeometry(1280, 720), false, "idempotent");
    assert.equal(mapper.setVideoGeometry(0, 0), false, "zero dimensions are ignored");
    assert.equal(mapper.videoWidth, 1280, "the ignored call did not clobber state");
    assert.equal(mapper.setDisplayGeometry(360, 640, ObjectFit.COVER), true);
    assert.equal(mapper.setDisplayGeometry(360, 640, ObjectFit.COVER), false);
    assert.equal(mapper.setDisplayGeometry(360, 640, ObjectFit.CONTAIN), true, "fit changes count");
});

test("describe reports the cropped fraction", () => {
    const mapper = new CoordinateMapper();
    mapper.setVideoGeometry(640, 480);
    mapper.setDisplayGeometry(360, 640, ObjectFit.COVER);
    const description = mapper.describe();
    assert.equal(description.video, "640x480");
    assert.ok(description.croppedFraction > 0.5, `expected heavy cropping, got ${description.croppedFraction}`);
});
