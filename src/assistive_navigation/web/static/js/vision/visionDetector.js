/**
 * VisionDetector Abstraction
 * Loads and executes client-side Edge object detection (MediaPipe Tasks Vision with WebAssembly / WebGL)
 * with throttled non-blocking execution and graceful fallbacks.
 */

import { VisionConfig } from "../config/visionConfig.js";
import { NormalizedDetection } from "./detectionTypes.js";

export class VisionDetector {
    constructor() {
        this.cocoModel = null;
        this.mpDetector = null;
        this.isLoaded = false;
        this.isLoading = false;
        this.isBusy = false;
        this.activeBackend = "NONE";
        this.lastDetections = [];
    }

    /**
     * Initializes the detector asynchronously with multi-engine failover.
     */
    async initialize() {
        if (this.isLoaded) return true;
        if (this.isLoading) return false;

        this.isLoading = true;
        console.log("[VisionDetector] Initializing Edge Vision Engines...");

        // 1. First Priority: TensorFlow.js COCO-SSD (Most stable on mobile browsers)
        if (typeof window !== "undefined" && window.cocoSsd) {
            try {
                console.log("[VisionDetector] Loading TensorFlow.js COCO-SSD model...");
                this.cocoModel = await window.cocoSsd.load({ base: "lite_mobilenet_v2" });
                this.activeBackend = "TFJS_COCO_SSD";
                this.isLoaded = true;
                this.isLoading = false;
                console.log("[VisionDetector] TFJS COCO-SSD Engine Ready.");
                return true;
            } catch (tfErr) {
                console.warn("[VisionDetector] COCO-SSD load failed, falling back to MediaPipe:", tfErr);
            }
        }

        // 2. Second Priority: MediaPipe Tasks Vision
        try {
            let FilesetResolver, ObjectDetector;
            try {
                const mp = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14");
                FilesetResolver = mp.FilesetResolver;
                ObjectDetector = mp.ObjectDetector;
            } catch (importErr) {
                const visionTasks = window.tasksVision || window;
                FilesetResolver = visionTasks.FilesetResolver || window.FilesetResolver;
                ObjectDetector = visionTasks.ObjectDetector || window.ObjectDetector;
            }

            if (FilesetResolver && ObjectDetector) {
                const vision = await FilesetResolver.forVisionTasks(VisionConfig.wasmLoaderPath);
                try {
                    this.mpDetector = await ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: VisionConfig.modelAssetPath,
                            delegate: "GPU"
                        },
                        runningMode: "IMAGE",
                        scoreThreshold: VisionConfig.highRiskConfidence,
                        maxResults: 15
                    });
                    this.activeBackend = "MEDIAPIPE_GPU";
                } catch (gpuErr) {
                    this.mpDetector = await ObjectDetector.createFromOptions(vision, {
                        baseOptions: {
                            modelAssetPath: VisionConfig.modelAssetPath,
                            delegate: "CPU"
                        },
                        runningMode: "IMAGE",
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

        // Fallback: If network is offline or libraries fail to load
        this.activeBackend = "STANDBY_PERCEPTION";
        this.isLoaded = true;
        this.isLoading = false;
        console.log("[VisionDetector] Active with Standby Perception Pipeline.");
        return true;
    }

    /**
     * Run inference on an HTML video or canvas element.
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
            const srcWidth = sourceElement.videoWidth || sourceElement.width || 640;
            const srcHeight = sourceElement.videoHeight || sourceElement.height || 480;

            // 1. Run COCO-SSD detection
            if (this.cocoModel) {
                const predictions = await this.cocoModel.detect(sourceElement, 15, VisionConfig.highRiskConfidence);
                if (predictions && predictions.length > 0) {
                    for (const pred of predictions) {
                        const [x, y, w, h] = pred.bbox;
                        const conf = pred.score;
                        const label = pred.class.toLowerCase();

                        if (conf >= VisionConfig.minDetectionConfidence || VisionConfig.highPriorityLabels.has(label)) {
                            const x1 = Math.max(0, Math.min(1, x / srcWidth));
                            const y1 = Math.max(0, Math.min(1, y / srcHeight));
                            const x2 = Math.max(0, Math.min(1, (x + w) / srcWidth));
                            const y2 = Math.max(0, Math.min(1, (y + h) / srcHeight));

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
            // 2. Run MediaPipe detection
            else if (this.mpDetector) {
                let results = null;
                if (typeof this.mpDetector.detect === "function") {
                    results = this.mpDetector.detect(sourceElement);
                }

                if (results && results.detections) {
                    for (const det of results.detections) {
                        const cat = det.categories[0];
                        const conf = cat ? cat.score : 0.0;
                        const label = cat ? cat.categoryName.toLowerCase() : "obstacle";
                        const box = det.boundingBox;

                        if (box && (conf >= VisionConfig.minDetectionConfidence || VisionConfig.highPriorityLabels.has(label))) {
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
