/**
 * Coordinate spaces and the transforms between them.
 *
 * Four spaces exist and they are NOT interchangeable:
 *
 *   1. MODEL space      - pixels inside the square, letterboxed model input.
 *                         Handled in vision/decoders.js.
 *   2. CAMERA space     - normalised 0..1 over the raw video frame
 *                         (`videoWidth` x `videoHeight`).
 *   3. DISPLAY space    - normalised 0..1 over the on-screen element, after
 *                         `object-fit` scaling/cropping and any CSS mirroring.
 *                         This is what overlay drawing needs.
 *   4. NAVIGATION space - normalised 0..1 over the region the user can actually
 *                         see, oriented so that x=0 is the user's real left.
 *                         This is what the risk and direction engines consume.
 *
 * Navigation space is derived from the *visible* region rather than the raw
 * frame on purpose. With `object-fit: cover` a phone crops the sides of a 4:3
 * sensor frame in portrait; an obstacle in that cropped strip is not in front of
 * the user's field of view and must not be allowed to swing a direction
 * decision.
 *
 * Pure geometry, no DOM. Importable in Node.
 */

/** Supported CSS object-fit behaviours for the preview element. */
export const ObjectFit = Object.freeze({
    COVER: "cover",
    CONTAIN: "contain",
    FILL: "fill"
});

function clamp01(v) {
    if (!(v > 0)) return 0;
    return v > 1 ? 1 : v;
}

/**
 * Resolve how a source frame is laid out inside a display box.
 *
 * @param {number} srcW
 * @param {number} srcH
 * @param {number} dstW
 * @param {number} dstH
 * @param {string} fit ObjectFit
 * @returns {{scaleX:number, scaleY:number, offsetX:number, offsetY:number,
 *            visible:{x1:number,y1:number,x2:number,y2:number}}}
 *          `visible` is the sub-rectangle of the source, in normalised source
 *          coordinates, that survives cropping.
 */
export function computeFitTransform(srcW, srcH, dstW, dstH, fit = ObjectFit.COVER) {
    const sw = Math.max(1, srcW || 1);
    const sh = Math.max(1, srcH || 1);
    const dw = Math.max(1, dstW || 1);
    const dh = Math.max(1, dstH || 1);

    let scaleX;
    let scaleY;
    if (fit === ObjectFit.FILL) {
        scaleX = dw / sw;
        scaleY = dh / sh;
    } else {
        const s = fit === ObjectFit.CONTAIN
            ? Math.min(dw / sw, dh / sh)
            : Math.max(dw / sw, dh / sh);
        scaleX = s;
        scaleY = s;
    }

    const renderedW = sw * scaleX;
    const renderedH = sh * scaleY;
    const offsetX = (dw - renderedW) / 2;
    const offsetY = (dh - renderedH) / 2;

    // Negative offset means the source overflows the box and is cropped.
    const cropLeftPx = offsetX < 0 ? -offsetX / scaleX : 0;
    const cropTopPx = offsetY < 0 ? -offsetY / scaleY : 0;

    return {
        scaleX,
        scaleY,
        offsetX,
        offsetY,
        renderedW,
        renderedH,
        visible: {
            x1: clamp01(cropLeftPx / sw),
            y1: clamp01(cropTopPx / sh),
            x2: clamp01((sw - cropLeftPx) / sw),
            y2: clamp01((sh - cropTopPx) / sh)
        }
    };
}

/**
 * Rotate a normalised box by a multiple of 90 degrees.
 *
 * Needed when the inference frame is explicitly rotated relative to the raw
 * sensor frame. Rotation is clockwise, matching how browsers report
 * `screen.orientation.angle`.
 *
 * @param {number[]} box [x1,y1,x2,y2] normalised
 * @param {number} degrees 0 | 90 | 180 | 270
 * @param {number[]} [out]
 */
export function rotateNormalisedBox(box, degrees, out = [0, 0, 0, 0]) {
    const [x1, y1, x2, y2] = box;
    const d = ((Math.round(degrees / 90) * 90) % 360 + 360) % 360;

    let a;
    let b;
    switch (d) {
        case 90:
            // (x,y) -> (1-y, x)
            a = [1 - y2, x1];
            b = [1 - y1, x2];
            break;
        case 180:
            a = [1 - x2, 1 - y2];
            b = [1 - x1, 1 - y1];
            break;
        case 270:
            // (x,y) -> (y, 1-x)
            a = [y1, 1 - x2];
            b = [y2, 1 - x1];
            break;
        default:
            a = [x1, y1];
            b = [x2, y2];
            break;
    }

    out[0] = Math.min(a[0], b[0]);
    out[1] = Math.min(a[1], b[1]);
    out[2] = Math.max(a[0], b[0]);
    out[3] = Math.max(a[1], b[1]);
    return out;
}

/** Mirror a normalised box horizontally. */
export function mirrorNormalisedBox(box, out = [0, 0, 0, 0]) {
    out[0] = 1 - box[2];
    out[1] = box[1];
    out[2] = 1 - box[0];
    out[3] = box[3];
    return out;
}

/**
 * Stateful mapper. One instance per app; geometry is refreshed whenever the
 * video metadata or the element size changes.
 */
export class CoordinateMapper {
    constructor() {
        this.videoWidth = 640;
        this.videoHeight = 480;
        this.displayWidth = 640;
        this.displayHeight = 480;
        this.objectFit = ObjectFit.COVER;
        /** CSS mirroring applied to the preview (front-camera convention). */
        this.mirrored = false;
        /** Extra rotation applied to the frame before display, in degrees. */
        this.rotationDegrees = 0;
        /** True when the active camera faces the user, so the scene is behind them. */
        this.facingUser = false;

        this._fit = computeFitTransform(640, 480, 640, 480, ObjectFit.COVER);
        this._scratchA = [0, 0, 0, 0];
        this._scratchB = [0, 0, 0, 0];
    }

    /** @returns {boolean} true when geometry changed and dependents should refresh. */
    setVideoGeometry(videoWidth, videoHeight) {
        if (!videoWidth || !videoHeight) return false;
        if (videoWidth === this.videoWidth && videoHeight === this.videoHeight) return false;
        this.videoWidth = videoWidth;
        this.videoHeight = videoHeight;
        this._recompute();
        return true;
    }

    /** @returns {boolean} true when geometry changed. */
    setDisplayGeometry(displayWidth, displayHeight, objectFit = this.objectFit) {
        const changed = displayWidth !== this.displayWidth
            || displayHeight !== this.displayHeight
            || objectFit !== this.objectFit;
        if (!changed || !displayWidth || !displayHeight) return false;
        this.displayWidth = displayWidth;
        this.displayHeight = displayHeight;
        this.objectFit = objectFit;
        this._recompute();
        return true;
    }

    setOrientation({ rotationDegrees = 0, mirrored = false, facingUser = false } = {}) {
        this.rotationDegrees = rotationDegrees;
        this.mirrored = mirrored;
        this.facingUser = facingUser;
    }

    _recompute() {
        this._fit = computeFitTransform(
            this.videoWidth,
            this.videoHeight,
            this.displayWidth,
            this.displayHeight,
            this.objectFit
        );
    }

    /** The sub-rectangle of the camera frame the user can actually see. */
    get visibleCameraRegion() {
        return this._fit.visible;
    }

    /** Aspect ratio of the visible region; used to scale vertical distance cues. */
    get visibleAspect() {
        const v = this._fit.visible;
        const w = (v.x2 - v.x1) * this.videoWidth;
        const h = (v.y2 - v.y1) * this.videoHeight;
        return h > 0 ? w / h : 1;
    }

    /**
     * CAMERA -> DISPLAY, in normalised display coordinates.
     * Use for overlay drawing. Includes CSS mirroring so what is drawn lines up
     * with what is rendered.
     */
    cameraToDisplay(box, out = [0, 0, 0, 0]) {
        const fit = this._fit;
        const src = this.rotationDegrees
            ? rotateNormalisedBox(box, this.rotationDegrees, this._scratchA)
            : box;

        const px1 = fit.offsetX + src[0] * this.videoWidth * fit.scaleX;
        const py1 = fit.offsetY + src[1] * this.videoHeight * fit.scaleY;
        const px2 = fit.offsetX + src[2] * this.videoWidth * fit.scaleX;
        const py2 = fit.offsetY + src[3] * this.videoHeight * fit.scaleY;

        out[0] = px1 / this.displayWidth;
        out[1] = py1 / this.displayHeight;
        out[2] = px2 / this.displayWidth;
        out[3] = py2 / this.displayHeight;

        if (this.mirrored) return mirrorNormalisedBox(out, out);
        return out;
    }

    /**
     * CAMERA -> NAVIGATION.
     *
     * Rescales into the visible region so that x=0 is the left edge of what the
     * user sees, then applies mirroring so x=0 is the user's physical left.
     * `visibility` reports how much of the box survives the crop; the risk
     * engine uses it to discount half-cropped objects.
     *
     * @returns {{box:number[], visibility:number, fullyOutside:boolean}}
     */
    cameraToNavigation(box) {
        const src = this.rotationDegrees
            ? rotateNormalisedBox(box, this.rotationDegrees, this._scratchA)
            : box;
        const vis = this._fit.visible;
        const spanX = vis.x2 - vis.x1;
        const spanY = vis.y2 - vis.y1;

        if (spanX <= 0 || spanY <= 0) {
            return { box: [0, 0, 0, 0], visibility: 0, fullyOutside: true };
        }

        const rawArea = Math.max(0, src[2] - src[0]) * Math.max(0, src[3] - src[1]);

        // Intersect with the visible region before rescaling.
        const ix1 = Math.max(src[0], vis.x1);
        const iy1 = Math.max(src[1], vis.y1);
        const ix2 = Math.min(src[2], vis.x2);
        const iy2 = Math.min(src[3], vis.y2);
        const iw = ix2 - ix1;
        const ih = iy2 - iy1;

        if (iw <= 0 || ih <= 0) {
            return { box: [0, 0, 0, 0], visibility: 0, fullyOutside: true };
        }

        const out = this._scratchB;
        out[0] = clamp01((ix1 - vis.x1) / spanX);
        out[1] = clamp01((iy1 - vis.y1) / spanY);
        out[2] = clamp01((ix2 - vis.x1) / spanX);
        out[3] = clamp01((iy2 - vis.y1) / spanY);

        const navBox = this.mirrored
            ? mirrorNormalisedBox(out, [0, 0, 0, 0])
            : [out[0], out[1], out[2], out[3]];

        return {
            box: navBox,
            visibility: rawArea > 0 ? Math.min(1, (iw * ih) / rawArea) : 0,
            fullyOutside: false
        };
    }

    /** DISPLAY -> CAMERA. Used to turn a screen tap into an image location. */
    displayToCamera(pointX, pointY) {
        const fit = this._fit;
        const dx = this.mirrored ? (1 - pointX) : pointX;
        const px = dx * this.displayWidth;
        const py = pointY * this.displayHeight;
        return {
            x: clamp01(((px - fit.offsetX) / fit.scaleX) / this.videoWidth),
            y: clamp01(((py - fit.offsetY) / fit.scaleY) / this.videoHeight)
        };
    }

    /** Snapshot for the diagnostics dashboard. */
    describe() {
        return {
            video: `${this.videoWidth}x${this.videoHeight}`,
            display: `${Math.round(this.displayWidth)}x${Math.round(this.displayHeight)}`,
            objectFit: this.objectFit,
            mirrored: this.mirrored,
            rotationDegrees: this.rotationDegrees,
            facingUser: this.facingUser,
            visible: this._fit.visible,
            croppedFraction: Number(
                (1 - (this._fit.visible.x2 - this._fit.visible.x1) * (this._fit.visible.y2 - this._fit.visible.y1)).toFixed(3)
            )
        };
    }
}
