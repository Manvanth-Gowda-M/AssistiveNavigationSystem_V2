/**
 * Camera Manager Module
 * Manages smartphone camera stream, facingMode "environment", permissions,
 * and real-time luminance / quality diagnostics.
 */

import { VisionConfig } from "../config/visionConfig.js";

export class CameraManager {
    constructor(videoElement) {
        this.video = videoElement;
        this.stream = null;
        this.isActive = false;
        this.diagnosticCanvas = document.createElement("canvas");
        this.diagnosticCtx = this.diagnosticCanvas.getContext("2d", { willReadFrequently: true });
        this.diagnosticCanvas.width = 64;
        this.diagnosticCanvas.height = 48;
    }

    /**
     * Requests rear camera with optimal resolution.
     */
    async startCamera() {
        if (this.isActive && this.stream) return true;

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error("Camera API unavailable in this browser context.");
        }

        const constraintCandidates = [
            // Strategy 1: Rear camera preferred with standard resolution
            {
                audio: false,
                video: {
                    facingMode: { ideal: "environment" },
                    width: { ideal: 640, min: 320 },
                    height: { ideal: 480, min: 240 }
                }
            },
            // Strategy 2: Simple rear camera
            {
                audio: false,
                video: { facingMode: { ideal: "environment" } }
            },
            // Strategy 3: Any available camera (front / webcam / USB)
            {
                audio: false,
                video: { facingMode: "user" }
            },
            // Strategy 4: Unconstrained camera
            {
                audio: false,
                video: true
            }
        ];

        let lastError = null;

        for (const constraints of constraintCandidates) {
            try {
                console.log("[CameraManager] Requesting camera with constraints:", JSON.stringify(constraints));
                this.stream = await navigator.mediaDevices.getUserMedia(constraints);
                this.video.srcObject = this.stream;
                this.video.setAttribute("playsinline", "true");
                this.video.setAttribute("autoplay", "true");
                this.video.muted = true;

                await new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        this.video.play().then(resolve).catch(resolve);
                    }, 1200);

                    this.video.onloadedmetadata = () => {
                        clearTimeout(timeout);
                        this.video.play().then(resolve).catch(resolve);
                    };
                });

                this.isActive = true;
                console.log("[CameraManager] Camera active:", this.video.videoWidth, "x", this.video.videoHeight);
                return true;
            } catch (err) {
                console.warn("[CameraManager] Constraint failed:", err.name, err.message);
                lastError = err;
            }
        }

        this.isActive = false;
        throw (lastError || new Error("Unable to access camera device."));
    }

    /**
     * Analyzes average luminance to detect low-light scenes.
     */
    checkQuality() {
        if (!this.isActive || !this.video.videoWidth) {
            return { isLowLight: false, brightness: 128 };
        }

        try {
            this.diagnosticCtx.drawImage(this.video, 0, 0, 64, 48);
            const imgData = this.diagnosticCtx.getImageData(0, 0, 64, 48);
            const data = imgData.data;
            let sumLuminance = 0;

            for (let i = 0; i < data.length; i += 4) {
                // ITU-R BT.601 luminance
                sumLuminance += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            }

            const avgBrightness = sumLuminance / (64 * 48);
            const isLowLight = avgBrightness < VisionConfig.lowLightBrightnessThreshold;

            return {
                isLowLight,
                brightness: Math.round(avgBrightness)
            };
        } catch (e) {
            return { isLowLight: false, brightness: 100 };
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
        this.isActive = false;
        console.log("[CameraManager] Camera stopped.");
    }
}
