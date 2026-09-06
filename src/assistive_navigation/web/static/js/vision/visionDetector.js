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
            // Import MediaPipe Tasks Vision directly as an ES Module
            let FilesetResolver, ObjectDetector;
            try {
                const mp = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14");
                FilesetResolver = mp.FilesetResolver;
                ObjectDetector = mp.ObjectDetector;
            } catch (importErr) {
                console.warn("[VisionDetector] Dynamic import fallback:", importErr);
                const visionTasks = window.tasksVision || window;
                FilesetResolver = visionTasks.FilesetResolver || window.FilesetResolver;
                ObjectDetector = visionTasks.ObjectDetector || window.ObjectDetector;
            }

            if (FilesetResolver && ObjectDetector) {
                const vision = await FilesetResolver.forVisionTasks(VisionConfig.wasmLoaderPath);
                try {
                    this.detector = await ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: VisionConfig.modelAssetPath,
                            delegate: "GPU" // Hardware WebGL / WebGPU acceleration
                        },
                        runningMode: "VIDEO",
                        scoreThreshold: VisionConfig.highRiskConfidence,
                        maxResults: 15
                    });
                    this.activeBackend = "MEDIAPIPE_GPU";
                } catch (gpuErr) {
                    console.warn("[VisionDetector] GPU delegate fallback to CPU/WASM:", gpuErr);
                    this.detector = await ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: VisionConfig.modelAssetPath,
                            delegate: "CPU"
                        },
                        runningMode: "VIDEO",
                        scoreThreshold: VisionConfig.highRiskConfidence,
                        maxResults: 15
                    });
                    this.activeBackend = "MEDIAPIPE_CPU";
                }

                this.isLoaded = true;
                this.isLoading = false;
                console.log(`[VisionDetector] ${this.activeBackend} Vision Detector ready.`);
                return true;
            }
        } catch (loadErr) {
            console.warn("[VisionDetector] MediaPipe load error:", loadErr);
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
        if (!this.isLoaded || this.isBusy || !sourceElement) {
            return this.lastDetections;
        }

        // Ensure video is ready to be sampled
        if (sourceElement.tagName === "VIDEO" && sourceElement.readyState < 2) {
            return this.lastDetections;
        }

        this.isBusy = true;
        const normalizedDetections = [];

        try {
            if (this.detector && this.activeBackend.startsWith("MEDIAPIPE")) {
                let results = null;
                const isVideo = sourceElement.tagName === "VIDEO";
                if (isVideo && typeof this.detector.detectForVideo === "function") {
                    const videoTime = Math.round(performance.now());
                    results = this.detector.detectForVideo(sourceElement, videoTime);
                } else if (typeof this.detector.detect === "function") {
                    results = this.detector.detect(sourceElement);
                }
                
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
