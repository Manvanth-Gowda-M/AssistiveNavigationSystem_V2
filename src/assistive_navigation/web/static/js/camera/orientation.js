/**
 * Orientation & Viewport Coordinate Normalization
 * Ensures bounding boxes correctly map across portrait/landscape orientations and mirror states.
 */

export class OrientationHelper {
    static getOrientation() {
        if (window.screen && window.screen.orientation) {
            return window.screen.orientation.type;
        }
        return window.innerWidth > window.innerHeight ? "landscape" : "portrait";
    }

    static isPortrait() {
        return window.innerHeight > window.innerWidth;
    }
}
