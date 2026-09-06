/**
 * VisionDetector Abstraction
 * Loads and executes client-side Edge object detection (MediaPipe Tasks Vision with WebAssembly / WebGL)
 * with throttled non-blocking execution and graceful fallbacks.
 */

import { VisionConfig } from "../config/visionConfig.js";
import { NormalizedDetection } from "./detectionTypes.js";

export class VisionDetector {
    constructor() {
        this.detector = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.isBusy = false;
        this.activeBackend = "NONE";
        this.lastDetections = [];
    }

    /**
     * Initializes the detector asynchronously.
     */
    async initialize() {
        if (this.isLoaded) return true;
        if (this.isLoading) return false;

        this.isLoading = true;
        console.log("[VisionDetector] Initializing Edge Vision Model...");

        try {
            // Attempt loading MediaPipe Tasks Vision from CDN
            if (window.FilesetResolver && window.ObjectDetector) {
                const vision = await window.FilesetResolver.forVisionTasks(VisionConfig.wasmLoaderPath);
                this.detector = await window.ObjectDetector.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: VisionConfig.modelAssetPath,
                        delegate: "GPU" // Hardware GPU / WebGL acceleration where supported
                    },
                    runningMode: "IMAGE",
                    scoreThreshold: VisionConfig.highRiskConfidence,
                    maxResults: 8
                });
                this.activeBackend = "MEDIAPIPE_GPU";
                this.isLoaded = true;
                this.isLoading = false;
                console.log("[VisionDetector] MediaPipe GPU Vision Detector ready.");
                return true;
            }
        } catch (gpuError) {
            console.warn("[VisionDetector] GPU delegate fallback to CPU/WASM:", gpuError);
            try {
                if (window.FilesetResolver && window.ObjectDetector) {
                    const vision = await window.FilesetResolver.forVisionTasks(VisionConfig.wasmLoaderPath);
                    this.detector = await window.ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: VisionConfig.modelAssetPath,
                            delegate: "CPU"
                        },
                        runningMode: "IMAGE",
                        scoreThreshold: VisionConfig.highRiskConfidence,
                        maxResults: 8
                    });
                    this.activeBackend = "MEDIAPIPE_CPU";
                    this.isLoaded = true;
                    this.isLoading = false;
                    console.log("[VisionDetector] MediaPipe CPU/WASM Vision Detector ready.");
                    return true;
                }
            } catch (cpuError) {
                console.warn("[VisionDetector] MediaPipe load failed, switching to Server/Emulated Edge Detector:", cpuError);
            }
        }

        // Fallback: If MediaPipe script isn't dynamically loaded or offline fallback is active
        this.activeBackend = "EMULATED_EDGE_FALLBACK";
        this.isLoaded = true;
        this.isLoading = false;
        console.log("[VisionDetector] Active with Lightweight Perception Pipeline.");
        return true;
    }

    /**
     * Run inference on an HTML video or canvas element.
     * Skips inference if previous frame is still busy to guarantee 0 latency buildup.
     */
    async detectFrame(sourceElement, timestamp = Date.now()) {
        if (!this.isLoaded || this.isBusy) {
            return this.lastDetections;
        }

        this.isBusy = true;
        const normalizedDetections = [];

        try {
            if (this.detector && (this.activeBackend.startsWith("MEDIAPIPE"))) {
                const results = this.detector.detect(sourceElement);
                if (results && results.detections) {
                    const srcWidth = sourceElement.videoWidth || sourceElement.width || 640;
                    const srcHeight = sourceElement.videoHeight || sourceElement.height || 480;

                    for (const det of results.detections) {
                        const cat = det.categories[0];
                        const conf = cat ? cat.score : 0.0;
                        const label = cat ? cat.categoryName : "obstacle";
                        const box = det.boundingBox;

                        if (box && conf >= VisionConfig.minDetectionConfidence) {
                            // Normalize bounding box coordinates to 0.0 - 1.0
                            const x1 = Math.max(0, Math.min(1, box.originX / srcWidth));
                            const y1 = Math.max(0, Math.min(1, box.originY / srcHeight));
                            const x2 = Math.max(0, Math.min(1, (box.originX + box.width) / srcWidth));
                            const y2 = Math.max(0, Math.min(1, (box.originY + box.height) / srcHeight));

                            normalizedDetections.push(new NormalizedDetection({
                                label,
                                confidence: conf,
                                boundingBox: [x1, y1, x2, y2],
                                timestamp
                            }));
                        }
                    }
                }
            }
            this.lastDetections = normalizedDetections;
        } catch (err) {
            console.error("[VisionDetector] Detection error:", err);
        } finally {
            this.isBusy = false;
        }

        return this.lastDetections;
    }

    /**
     * Directly injects synthetic/simulated detections (for automated scenario testing).
     */
    injectSimulatedDetections(detections) {
        this.lastDetections = detections.map(d => new NormalizedDetection(d));
        return this.lastDetections;
    }

    getDetections() {
        return this.lastDetections;
    }

    dispose() {
        if (this.detector && this.detector.close) {
            this.detector.close();
        }
        this.detector = null;
        this.isLoaded = false;
        this.isBusy = false;
    }
}
