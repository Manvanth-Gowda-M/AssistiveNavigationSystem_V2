# Real-Time Architecture Audit (pre-upgrade baseline)

Audit of the browser walking-assistance app as it existed before the real-time
upgrade. Scope is the static web client under
`src/assistive_navigation/web/static/`. The Python pipeline under
`src/assistive_navigation/{camera,depth,detection,tracking,navigation,alerts,audio}`
is a separate desktop/offline research pipeline and is **not** on the smartphone
demo path; it is left untouched.

## 1. Inventory

| Aspect | Finding |
| --- | --- |
| Framework | None. Vanilla ES modules, no framework. |
| Build system | None. `index.html` loads `./js/app.js` as `type="module"`. |
| Deployment | `.github/workflows/deploy-pages.yml` uploads `src/assistive_navigation/web/static` directly to GitHub Pages. |
| Camera | `camera/cameraManager.js`, `getUserMedia`, `facingMode: environment`, ideal 640x480, 4-step constraint fallback ladder. |
| Detector | `vision/visionDetector.js`. Primary: TF.js `coco-ssd` (`lite_mobilenet_v2`) from CDN. Fallback: MediaPipe `ObjectDetector` with EfficientDet-Lite0 **float16** `.tflite`. Last resort: `STANDBY_PERCEPTION` (no detector at all). |
| Model format | TFJS graph model (CDN) / TFLite float16 (Google Storage). |
| Inference runtime | TF.js default backend (WebGL on most phones) or MediaPipe WASM/GPU delegate. |
| Inference resolution | Uncontrolled. The full `<video>` element is handed to the detector; internal resize is opaque. `VisionConfig.inputResolution = 384x384` is declared but **never referenced anywhere**. |
| Inference frequency | Fixed 66 ms gate (`inferenceIntervalMs`) plus `!detector.isBusy`. No adaptation. |
| Camera resolution used for inference | Same as preview (no low-res inference stream). |
| Tracking | `vision/temporalTracker.js`. Greedy per-track IoU match, threshold 0.20, no centroid fallback, no motion model while coasting. |
| Speech | `audio/speechEngine.js` + `speechQueue.js` + `voiceManager.js`, Web Speech API. |
| State management | `navigation/stateMachine.js`, 12 states, no persistence of health. |
| Rendering loop | `requestAnimationFrame` loop in `app.js` that `await`s inference inline. |
| Error handling | `try/catch` that logs and returns the previous detection list. |

## 2. Confirmed defects, mapped to reported symptoms

### 2.1 Inference runs on the main thread
`app.js` `startLoop()` does `await this.detector.detectFrame(...)` inside the
`requestAnimationFrame` callback. TF.js `coco-ssd` `detect()` performs GPU
readback (`dataSync`-class operations) and NMS on the main thread. Every
inference therefore competes with layout, canvas overlay painting and speech
scheduling.

*Explains:* "freezes", "becomes slow while walking", "UI stutter".

### 2.2 No frame downscale before inference
The raw `<video>` element is the inference source. On a phone that negotiates
1280x720 (strategy 2 or 4 in the constraint ladder always can), the detector
resizes a 720p frame every cycle.

*Explains:* latency spikes, thermal throttling, battery drain.

### 2.3 Detection thresholds are pathologically low
`minDetectionConfidence: 0.15`, `highRiskConfidence: 0.12`,
`temporalConfirmFrames: 1`. A track is confirmed on its **first** frame, so
temporal confirmation is effectively disabled and every 0.12-confidence blob
immediately participates in navigation decisions.

*Explains:* "inconsistent directions", "repeatedly changes direction",
"speaks too much".

### 2.4 The barrier detector is a false-STOP generator
`vision/barrierDetector.js` declares a `wall` at confidence 0.85 spanning
`[0.20, 0.15, 0.80, 0.90]` (58% of the frame) whenever the centre 50% of a
32x24 thumbnail has luminance std-dev < 15.5. A plain floor, a road surface,
the sky, or a blank wall 10 m away all satisfy this. Because the box is huge,
`RiskEngine.estimateDistanceCategory` returns `VERY_CLOSE`, which the decision
engine turns into an unconditional `STOP`.

*Explains:* spurious stops, "does not understand whether objects actually
block the path".

### 2.5 Silent permanent failure
`detectFrame` catches every error and returns `this.lastDetections`. If the
WebGL context is lost (common on Android after backgrounding) the app keeps
serving the last successful frame's detections forever, with no state change
and no user notification.

*Explains:* "stops detecting temporarily", "misses objects", and is a safety
defect in its own right.

### 2.6 Hysteresis does not actually hold
`NavigationDecisionEngine.applyHysteresis` only suppresses a **strictly
opposite** direction, and it resets `decisionHoldCount` to the full window on
every accepted decision. Churn between `MOVE_SLIGHTLY_RIGHT`, `CAUTION`,
`STOP` and `CONTINUE` is unrestricted, and there is no risk-margin
requirement for a flip.

*Explains:* "repeatedly changes direction".

### 2.7 Direction default is arbitrary
When flanks are within 0.10 of each other the engine emits
`MOVE_SLIGHTLY_RIGHT` with no basis. There is no check that the recommended
side is itself free of high-risk obstacles.

### 2.8 Free space is inferred only from bounding boxes
`GeometryUtils.estimateFlankClearance` sums horizontal box overlap on each
flank. Vertical position, ground contact and unoccupied-column structure are
ignored, so a bus 40 m up the road blocks a flank as much as a bin at your feet.

### 2.9 Corridor is a fixed rectangle
`NavigationConfig.corridor` is a constant `0.28 - 0.72` band. It does not adapt
to device orientation, camera pitch or the observed ground region.

### 2.10 Speech is long and globally rate-limited
`generatePhrase` emits `"Chair ahead. Move slightly right."` rather than
`"Move slightly right."`. `lastSpokenTime` is a single global timestamp updated
at **dequeue** time, not at utterance end, so a long utterance still frees the
gate early while any speech at all blocks unrelated higher-value speech for
900 ms. Live narration mode fires a scene description every 3.8 s.

*Explains:* "speaks too much".

### 2.11 No health monitoring, watchdog, recovery or degradation
There is no measurement of dropped frames, p95 latency, consecutive inference
failures, or last-successful-inference time; no worker to restart; no runtime
backend fallback; no degradation ladder. `PerformanceTracker.getFps()` counts
`requestAnimationFrame` ticks but the debug panel labels it "Inference FPS".

### 2.12 No model caching or warm-up
`coco-ssd.load()` and the MediaPipe `.tflite` fetch rely purely on the HTTP
cache. There are no warm-up passes, so the first few live frames are the
slowest frames of the session — exactly when the user starts walking.

### 2.13 Per-frame allocations
`cameraManager.checkQuality()` and `barrierDetector.detectWallBarrier()` each
call `getImageData` every frame (two fresh `ImageData` allocations per frame),
and `checkQuality` runs on **every rAF tick**, not per inference. `smoothBox`
allocates a new 4-element array per track per update.

### 2.14 Coordinate mapping is unverified
`camera/orientation.js` exposes only `getOrientation()`/`isPortrait()` and is
never imported. Nothing maps between camera space, model input space
(letterboxing), CSS display space (`object-fit`) and navigation space. The
debug canvas is sized to `videoWidth/videoHeight` and stretched by CSS, so
overlay boxes are only accidentally aligned.

## 3. What is worth preserving

These components are sound and are kept (refactored, not rewritten):

- `camera/cameraManager.js` constraint fallback ladder — good defensive design.
- `audio/voiceManager.js` voice discovery and preference logic.
- `audio/speechQueue.js` priority queue with emergency pre-emption.
- `navigation/stateMachine.js` explicit state model.
- `ui/assistanceControls.js` accessible controls and keyboard shortcuts.
- `utils/geometry.js` IoU and corridor-overlap primitives.
- `ui/scenarioRunner.js` synthetic scenario definitions (reused for automated tests).
- The TF.js COCO-SSD path — retained as the guaranteed-working fallback adapter.

## 4. Hard constraints the upgrade must respect

1. **No bundler.** The deployment uploads the `static/` directory verbatim, so
   everything must run as native ES modules plus classic worker scripts.
2. **No cross-origin isolation on GitHub Pages.** `Cross-Origin-Opener-Policy`
   and `Cross-Origin-Embedder-Policy` cannot be set, so `SharedArrayBuffer` is
   unavailable and ONNX Runtime WASM **multi-threading will not work** on the
   Pages deployment. Thread count must be gated on
   `self.crossOriginIsolated`, and single-thread SIMD must be a first-class
   path rather than a degraded afterthought.
3. **Camera requires a secure context.** localhost or HTTPS only.
4. Everything on the guidance path stays local; no frame ever leaves the device.

## 5. Target architecture

```
camera (preview res)          camera (inference res, offscreen)
      |                                   |
      v                                   v
  <video> render                    FramePipeline
                                     (preallocated OffscreenCanvas,
                                      ImageBitmap transfer)
                                          |
                                    FrameScheduler
                                (adaptive rate, single in-flight,
                                 newest-frame-only, drop counting)
                                          |
                                    InferenceClient
                                  /                \
                       Worker (ORT-Web)         Main thread
                       ONNX YOLO adapter        MediaPipe / TFJS adapters
                                  \                /
                                       Detections
                                          |
                            +-------------+-------------+
                            |                           |
                       ObjectTracker              FreeSpaceEstimator
                    (IoU + centroid, velocity,   (column occupancy grid,
                     smoothing, two-tier,         ground band, optional
                     scene-change decay)          adaptive segmentation)
                            |                           |
                            +-------------+-------------+
                                          |
                                    CorridorModel (dynamic)
                                          |
                                     RiskEngine
                                          |
                                    PathScoreEngine
                                          |
                                  DecisionEngine (hysteresis,
                                   multi-object, confidence)
                                          |
                                    SpeechManager
                                          |
                                      user audio

parallel: VisionHealthMonitor | VisionWatchdog | RecoveryManager
          BackendSelector | ModelCache | DegradationController
```

## 6. Priority order applied to every decision in this upgrade

1. Safety
2. Continuous operation
3. Low latency
4. Stable tracking
5. Free-space understanding
6. Correct direction
7. Speech quality
8. Model accuracy
9. UI polish
