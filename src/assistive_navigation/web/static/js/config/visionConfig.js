/**
 * Vision Configuration Parameters
 * Controls detection models, confidence thresholds, frame rates, and resolution.
 */

export const VisionConfig = {
    // MediaPipe Tasks Vision CDN & Model configurations
    wasmLoaderPath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
    fallbackModelPath: "https://storage.googleapis.com/mediapipe-models/object_detector/ssd_mobilenet_v2/float16/1/ssd_mobilenet_v2.tflite",

    // Throttling & Frame Skipping
    targetInferenceFps: 15,          // 10-20 FPS target for edge AI inference
    inferenceIntervalMs: 66,        // 1000 / 15 = ~66ms between inference frames
    inputResolution: { width: 384, height: 384 }, // Model-appropriate resized frame

    // Confidence Thresholds
    minDetectionConfidence: 0.25,   // Floor confidence for object acceptance (tuned for indoor lighting)
    highRiskConfidence: 0.20,       // Lower floor for high-risk hazards (vehicles, obstacles)
    temporalConfirmFrames: 1,       // 1-frame instant reactivity for immediate obstacle alerts
    trackTimeoutFrames: 10,         // Frames to coast before dropping lost track

    // Camera Quality Diagnostics
    lowLightBrightnessThreshold: 28, // Below this average luminance triggers "Low visibility"
    motionBlurGradientThreshold: 12, // Below this Laplacian variance indicates blur

    // High Priority Semantic Categories
    highPriorityLabels: new Set([
        "person", "bicycle", "car", "motorcycle", "bus", "truck",
        "stairs", "door", "stop sign", "traffic light", "fire hydrant",
        "laptop", "tv", "cell phone"
    ]),

    mediumPriorityLabels: new Set([
        "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "suitcase", "backpack", "umbrella", "bench",
        "keyboard", "mouse", "bottle", "cup", "book"
    ])
};
