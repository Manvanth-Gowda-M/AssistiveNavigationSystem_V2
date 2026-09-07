/**
 * Camera manager.
 *
 * The constraint fallback ladder from the pre-upgrade build is kept - it is good
 * defensive design and it works on awkward devices. What changed:
 *
 *  - The preview stream is requested at a *sensible* resolution for display, and
 *    is never used as the inference source. Inference gets its own downscaled,
 *    letterboxed stream from `FramePipeline`.
 *  - Frame-quality sampling has moved out. This class no longer calls
 *    `getImageData` on every animation frame.
 *  - Orientation, facing mode and mirroring are reported, so the coordinate
 *    mapper can guarantee that "left" means left.
 */

import { VisionConfig } from "../config/visionConfig.js";

export const CameraFacing = Object.freeze({
    ENVIRONMENT: "environment",
    USER: "user",
    UNKNOWN: "unknown"
});

export class CameraManager {
    /**
     * @param {HTMLVideoElement} videoElement
     * @param {{onGeometryChange?:Function, onTrackEnded?:Function}} [opts]
     */
    constructor(videoElement, { onGeometryChange = () => {}, onTrackEnded = () => {} } = {}) {
        this.video = videoElement;
        this.onGeometryChange = onGeometryChange;
        this.onTrackEnded = onTrackEnded;

        this.stream = null;
        this.isActive = false;
        this.facing = CameraFacing.UNKNOWN;
        this.settings = null;
        this.capabilities = null;
        this.usedStrategy = null;
        this.lastError = null;

        this._orientationListener = null;
        this._resizeListener = null;
        this.orientationAngle = 0;
    }

    /**
     * Constraint ladder, most-preferred first.
     *
     * `facingMode: {ideal: 'environment'}` rather than `{exact: ...}` on purpose:
     * `exact` fails outright on laptops and on some Android devices that report
     * their rear camera oddly, and a working front camera plus a clear warning
     * beats no camera at all.
     */
    _constraintLadder() {
        const p = VisionConfig.preview;
        return [
            {
                label: "rear-preferred-720p",
                constraints: {
                    audio: false,
                    video: {
                        facingMode: { ideal: "environment" },
                        width: { ideal: p.idealWidth, min: p.minWidth },
                        height: { ideal: p.idealHeight, min: p.minHeight },
                        frameRate: { ideal: p.idealFrameRate, min: 15 }
                    }
                }
            },
            {
                label: "rear-simple",
                constraints: { audio: false, video: { facingMode: { ideal: "environment" } } }
            },
            {
                label: "front",
                constraints: { audio: false, video: { facingMode: "user" } }
            },
            {
                label: "any",
                constraints: { audio: false, video: true }
            }
        ];
    }

    async start() {
        if (this.isActive && this.stream) return { ok: true, reused: true };

        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error(
                "Camera API unavailable. A secure context (HTTPS or localhost) is required."
            );
        }

        let lastError = null;

        for (const strategy of this._constraintLadder()) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(strategy.constraints);
                await this._attach(stream, strategy.label);
                return { ok: true, strategy: strategy.label, facing: this.facing };
            } catch (err) {
                lastError = err;
                this.lastError = `${err.name}: ${err.message}`;
                // A hard permission denial will not be fixed by a looser
                // constraint, so stop rather than prompting three more times.
                if (err.name === "NotAllowedError" || err.name === "SecurityError") break;
            }
        }

        this.isActive = false;
        throw (lastError || new Error("Unable to access any camera device."));
    }

    async _attach(stream, strategyLabel) {
        this.stream = stream;
        this.usedStrategy = strategyLabel;

        const [track] = stream.getVideoTracks();
        if (track) {
            this.settings = typeof track.getSettings === "function" ? track.getSettings() : null;
            this.capabilities = typeof track.getCapabilities === "function" ? track.getCapabilities() : null;
            const reported = this.settings?.facingMode;
            this.facing = reported === "environment"
                ? CameraFacing.ENVIRONMENT
                : (reported === "user" ? CameraFacing.USER : (strategyLabel === "front" ? CameraFacing.USER : CameraFacing.UNKNOWN));

            // A track can end without an error: unplugged webcam, OS camera
            // switch, or the app being backgrounded on some Android builds.
            track.addEventListener("ended", () => {
                this.isActive = false;
                this.onTrackEnded({ reason: "track-ended" });
            });
        }

        this.video.srcObject = stream;
        this.video.setAttribute("playsinline", "true");
        this.video.muted = true;

        await this._waitForFirstFrame();

        this.isActive = true;
        this._installGeometryListeners();
        this._emitGeometry();
        return true;
    }

    /**
     * Wait until the element actually has frame dimensions.
     *
     * `loadedmetadata` is not sufficient on iOS Safari, which can report the
     * event with `videoWidth === 0`. `requestVideoFrameCallback` is the reliable
     * signal where it exists; the timeout is the backstop so start-up cannot
     * hang forever.
     */
    _waitForFirstFrame(timeoutMs = 6000) {
        return new Promise((resolve) => {
            let settled = false;
            const done = (how) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                resolve(how);
            };

            const timer = setTimeout(() => done("timeout"), timeoutMs);

            const check = () => {
                if (this.video.videoWidth > 0 && this.video.videoHeight > 0) done("dimensions");
            };

            this.video.addEventListener("loadedmetadata", check, { once: true });
            this.video.addEventListener("loadeddata", check, { once: true });

            if (typeof this.video.requestVideoFrameCallback === "function") {
                this.video.requestVideoFrameCallback(() => done("rvfc"));
            }

            this.video.play().then(check).catch(() => {
                // Autoplay rejection: the stream is still attached and will play
                // once the element is visible. Do not treat as fatal.
                check();
            });

            check();
        });
    }

    _installGeometryListeners() {
        this._removeGeometryListeners();

        const handler = () => {
            this.orientationAngle = this._readOrientationAngle();
            this._emitGeometry();
        };

        this._orientationListener = handler;
        this._resizeListener = handler;

        if (screen?.orientation?.addEventListener) {
            screen.orientation.addEventListener("change", handler);
        } else {
            window.addEventListener("orientationchange", handler);
        }
        window.addEventListener("resize", handler);
        this.video.addEventListener("resize", handler);
    }

    _removeGeometryListeners() {
        if (!this._orientationListener) return;
        if (screen?.orientation?.removeEventListener) {
            screen.orientation.removeEventListener("change", this._orientationListener);
        } else {
            window.removeEventListener("orientationchange", this._orientationListener);
        }
        window.removeEventListener("resize", this._resizeListener);
        this.video.removeEventListener("resize", this._resizeListener);
        this._orientationListener = null;
        this._resizeListener = null;
    }

    _readOrientationAngle() {
        if (screen?.orientation && Number.isFinite(screen.orientation.angle)) return screen.orientation.angle;
        if (Number.isFinite(window.orientation)) return ((window.orientation % 360) + 360) % 360;
        return 0;
    }

    _emitGeometry() {
        this.onGeometryChange(this.geometry);
    }

    /**
     * Everything the coordinate mapper needs.
     *
     * `rotationDegrees` is 0 because every browser we target delivers frames in
     * display orientation - `videoWidth`/`videoHeight` already swap in portrait.
     * The field exists so that a device found to behave otherwise can be
     * corrected in one place rather than by patching the overlay.
     */
    get geometry() {
        const rect = this.video.getBoundingClientRect?.() || { width: 0, height: 0 };
        return {
            videoWidth: this.video.videoWidth || 0,
            videoHeight: this.video.videoHeight || 0,
            displayWidth: rect.width || this.video.clientWidth || 0,
            displayHeight: rect.height || this.video.clientHeight || 0,
            orientationAngle: this.orientationAngle,
            rotationDegrees: 0,
            // Only a user-facing camera is mirrored, matching the CSS transform
            // applied to the preview in that case.
            mirrored: this.facing === CameraFacing.USER,
            facingUser: this.facing === CameraFacing.USER,
            portrait: (this.video.videoHeight || 0) > (this.video.videoWidth || 0)
        };
    }

    /** True when the element currently has a decodable frame. */
    get hasFrame() {
        return this.isActive && this.video.readyState >= 2 && this.video.videoWidth > 0;
    }

    /** Torch, where the platform exposes it. Useful in dim indoor demos. */
    async setTorch(on) {
        const [track] = this.stream?.getVideoTracks() || [];
        if (!track?.applyConstraints) return false;
        if (!this.capabilities || !("torch" in this.capabilities)) return false;
        try {
            await track.applyConstraints({ advanced: [{ torch: Boolean(on) }] });
            return true;
        } catch {
            return false;
        }
    }

    stop() {
        this._removeGeometryListeners();
        if (this.stream) {
            for (const track of this.stream.getTracks()) track.stop();
            this.stream = null;
        }
        if (this.video) this.video.srcObject = null;
        this.isActive = false;
    }

    describe() {
        return {
            active: this.isActive,
            facing: this.facing,
            strategy: this.usedStrategy,
            resolution: `${this.video.videoWidth || 0}x${this.video.videoHeight || 0}`,
            frameRate: this.settings?.frameRate ? Math.round(this.settings.frameRate) : null,
            deviceLabel: this.settings?.deviceId ? "granted" : "unknown",
            orientationAngle: this.orientationAngle,
            lastError: this.lastError
        };
    }
}

export default CameraManager;
