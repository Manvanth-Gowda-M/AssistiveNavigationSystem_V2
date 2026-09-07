# Real-Time Browser Pipeline

Reference for the smartphone walking-assistance client that lives in
`src/assistive_navigation/web/static/`. This is a separate system from the
Python desktop pipeline documented in `docs/architecture.md`; the two share
design ideas but no code.

For what the previous version did and why it was changed, see
[`realtime_architecture_audit.md`](realtime_architecture_audit.md).

---

## 1. Shape of the system

```
                      CAMERA
              (preview resolution, on screen)
                         │
        ┌────────────────┴────────────────┐
        │                                 │
   <video> render                  FramePipeline
   (never blocked)          preallocated OffscreenCanvas,
                            letterbox to model input,
                            pooled Float32Array tensor
                                          │
                                   FrameScheduler
                        adaptive rate · single in flight ·
                        newest frame only · drop accounting
                                          │
                                   InferenceClient
                     ┌────────────────────┴────────────────────┐
                     │                                         │
          Worker (ONNX Runtime Web)                   Main thread (recovery)
          WebGPU or WASM SIMD                         MediaPipe · TensorFlow.js
                     └────────────────────┬────────────────────┘
                                          │
                                    detections
                                  (camera space)
                                          │
                                  CoordinateMapper
                              camera → navigation space
                                          │
                     ┌────────────────────┴────────────────────┐
                     │                                         │
               ObjectTracker                          FreeSpaceEstimator
        IoU + centroid association,          column occupancy grid, ground band,
        velocity, smoothing, coasting,       optional ground-freeness refinement
        two-tier (normal + emergency)
                     └────────────────────┬────────────────────┘
                                          │
                                    CorridorModel
                                          │
                                     RiskEngine
                                          │
                                  PathScoreEngine
                                  left / centre / right
                                          │
                                  DecisionEngine
                        hysteresis · confidence · emergency
                                          │
                                  SpeechManager
                          priority · dedup · emergency bypass
                                          │
                                     USER AUDIO

alongside:  VisionHealthMonitor · VisionWatchdog · RecoveryManager
            DegradationController · BackendSelector · ModelCache
```

## 2. Modules

| Path | Responsibility |
| --- | --- |
| `js/app.js` | Orchestrator. Owns the render loop, wiring and lifecycle. |
| `js/camera/cameraManager.js` | `getUserMedia` with a constraint fallback ladder; reports geometry, facing mode and orientation. |
| `js/vision/framePipeline.js` | Letterboxed downscale into a reused canvas; pooled tensor buffers; shared analysis thumbnail. |
| `js/vision/frameScheduler.js` | When to run inference, and the p95-driven rate controller. |
| `js/vision/inferenceClient.js` | Backend/model selection, warm-up gating, single-in-flight submission, recovery entry points. |
| `js/workers/inferenceWorker.js` | ES-module worker running ONNX Runtime Web and output decoding. |
| `js/vision/adapters/*.js` | One adapter per runtime: ORT worker, MediaPipe, TensorFlow.js. |
| `js/vision/decoders.js` | Letterbox geometry, YOLO raw and end-to-end decode, class-aware NMS. |
| `js/vision/backendSelector.js` | Backend plan, ORT environment configuration, warm-up verdict. |
| `js/vision/modelCache.js` | IndexedDB cache for model bytes. |
| `js/vision/coordinateMapper.js` | Camera / display / navigation coordinate transforms. |
| `js/vision/tracker.js` | Temporal tracking and the emergency channel. |
| `js/vision/frameQuality.js` | Brightness, contrast, blur; rolling degradation state. |
| `js/vision/sceneChange.js` | Scene change and camera-turn detection. |
| `js/perception/freeSpace.js` | Column occupancy grid, sector clearance, widest passable gap. |
| `js/perception/segmentationStage.js` | Ground-freeness refinement plus its load governor. |
| `js/navigation/corridorModel.js` | Adaptive walking corridor. |
| `js/navigation/riskEngine.js` | Proximity bands and 0–100 collision risk. |
| `js/navigation/pathScoring.js` | Left / centre / right path scores. |
| `js/navigation/decisionEngine.js` | One action, held stably. |
| `js/audio/speechPolicy.js` | What to say, and whether to say anything. |
| `js/audio/speechManager.js` | Web Speech plumbing, queue, interruption. |
| `js/health/*.js` | Health monitor, watchdog, recovery ladder, degradation ladder. |
| `js/ui/*.js` | Status view, controls, overlay, diagnostics, benchmark, demo mode, scenario replay. |
| `js/testing/*.js` | Scripted scenarios and the deterministic pipeline harness. |

Every module on the reasoning path is free of DOM access at import time, which is
what allows the whole decision chain to be tested under Node. `tests/js/integrity.test.mjs`
enforces that.

## 3. The two rules that matter most

### The render loop never awaits inference

`app.js` re-schedules `requestAnimationFrame` on the first line of every
callback and fires inference without awaiting it. The camera preview and the UI
therefore cannot be blocked by the model. In the previous build the loop awaited
`detect()` inline, which is why walking made it stutter.

### There is no frame queue

`FrameScheduler.shouldRun()` refuses to submit while inference is in flight or
while no tensor buffer is free. `WorkerDetectorAdapter.detect()` rejects rather
than queues if called anyway, and the worker asserts the same invariant. Skipped
frames are counted by reason and shown on the diagnostics panel.

Rate limiting is deliberately *not* counted as a dropped frame: running inference
at 12 FPS against a 60 FPS render loop is the intended behaviour, not a symptom.

## 4. Allocation and memory

| Concern | Approach |
| --- | --- |
| Inference canvas | One `OffscreenCanvas`, created once, resized only when the input size changes. |
| Tensor buffers | Pool of three `Float32Array`s, transferred to the worker and transferred back. The pool cannot leak: the worker returns the buffer on both the success and the error path, and a lost buffer is accounted for explicitly. |
| Analysis thumbnail | One 64×48 canvas plus one persistent `Uint8Array` luminance plane, shared by the quality, scene-change and refinement stages. |
| Output tensor | The worker allocates a reusable output tensor after the first run reveals the dims, and feeds it back to ORT via `fetches`. If an execution provider refuses preallocated outputs it falls back once, permanently, rather than retrying every frame. |
| Decoder scratch | Grow-only typed arrays in `DetectionDecoder`; no per-frame allocation at steady state. |
| Track history | Capped at `VisionConfig.tracking.historyLength` entries per track. |
| Heap watch | `VisionHealthMonitor.sampleMemory()` samples `performance.memory` where available and warns on sustained growth. Where it is unavailable that is reported, not assumed to be zero. |

One unavoidable allocation remains: `getImageData` returns a fresh `ImageData` on
every inference, because the 2D canvas API has no read-into-existing-buffer form.
It is a fixed 320×320×4 ≈ 410 KB short-lived buffer, it does not grow, and it is
the only one. Eliminating it would require a WebGL `readPixels` path in the
worker; that is a deliberate deferral, not an oversight.

## 5. Backend selection

1. Probe capabilities (`config/deviceProfiles.js`).
2. Build an ordered plan (`vision/backendSelector.js`):
   WebGPU → WASM SIMD → WASM → MediaPipe GPU → MediaPipe CPU → TensorFlow.js.
3. For each backend, try the compatible models cheapest-first.
4. For each candidate: create session → warm up → measure → judge.
5. Accept the first candidate that passes the gate.

The gate has three independent conditions:

- it produced output at all;
- the output was structurally sane (`checkTensorSanity` rejects NaN, Infinity and
  all-zero tensors, which is how an "available but broken" WebGPU driver
  presents);
- p95 warm-up latency is inside `VisionConfig.warmup.maxAcceptableLatencyMs`.

**WebGPU being available is never treated as WebGPU being faster.** It is
measured like everything else, and rejected if it does not perform.

### WASM threads

Threads need `SharedArrayBuffer`, which needs cross-origin isolation, which
GitHub Pages cannot provide (no COOP/COEP headers). `detectWasmThreads()` checks
`self.crossOriginIsolated` rather than assuming, and single-threaded SIMD is
treated as a first-class path. Quantised (uint8) weights are preferred on that
path, matching ONNX Runtime's own guidance for CPU execution.

## 6. Perception detail

### Distance is approximate, and labelled as such

Bands are `VERY_NEAR`, `NEAR`, `MEDIUM`, `FAR`, `UNKNOWN`. The primary cue is the
box's bottom edge (ground contact); apparent area is secondary. Either can promote
a band, because a bus filling the frame without a visible base is still near.
Nothing in the system claims a distance in metres, and nothing presents one to
the user.

Where ground contact is not visible at all (`bottomY` above the ground band) the
vertical cue is unusable — it would read "fully open" for a wall two feet away —
so apparent size takes over.

### Free space is a column grid, not a bounding-box tally

24 columns across the navigation width. Each column carries a **free depth** in
0–1: how far forward it is walkable. An obstacle reduces the depth of the columns
it covers, in proportion to its confidence and hazard class, with a small lateral
margin so the system never recommends clipping past something by two centimetres.

Sector clearance weights the *worst* column heavily (`minWeight` 0.55 against
`meanWeight` 0.45), because an otherwise-clear sector with one blocked column is
not clear. Separately, each sector's **widest contiguous passable run** is
measured against a shoulder width; a sector narrower than that is unusable no
matter how empty it looks on average.

### The refinement stage replaces the old barrier detector

`GroundFreenessEstimator` walks upward from the bottom of the analysis thumbnail
per column and stops at the first strong appearance discontinuity relative to a
bootstrapped floor model. That row is where the walkable floor ends. It catches
kerbs, steps, pallets, walls and closed doors — the obstacles COCO has no class
for.

It costs roughly 0.05 ms and is fused into the detection grid with `min`
semantics: either source may veto a column, neither may unblock one.
`AdaptiveSegmentationController` runs it only when it can change an outcome
(obstacle present, direction uncertain, scene changed) plus a low-rate periodic
refresh, measures its own cost, and disables itself if it exceeds its budget or
starts dominating a struggling pipeline.

This replaced `barrierDetector.js`, which declared a frame-filling "wall" at 0.85
confidence whenever the centre of the image was low-texture. That fired on plain
floors, roads and sky, and every one of those became a false STOP.

### The corridor does not chase openings

The corridor's half-width adapts to clutter and its vertical extent adapts to how
much open floor is visible (a proxy for camera pitch). Its **centre does not
move**.

An earlier version steered the centre towards the widest free-space gap. Because
the corridor defines the LEFT / CENTRE / RIGHT sector boundaries, moving it
towards an opening shrank the sector on that side, lowered that sector's passable
width, and inverted the path scores — a feedback loop that produced exactly the
left/right oscillation this upgrade set out to remove. Choosing a side is the
decision engine's job. `tests/js/freeSpace.test.mjs` guards against the
regression.

## 7. Risk and decisions

Risk is **multiplicative** over normalised factors: proximity, path overlap,
ground relevance, confidence, tracking stability, visibility after crop, approach
rate, lateral convergence and hazard class. If any single factor says "this
cannot hit me", the product collapses. An additive model lets a confident, large,
irrelevant object accumulate a dangerous-looking score, which is how you end up
stopping for a parked car across the road.

Thresholds (`NavigationConfig.risk`): notice 26, avoid 46, critical 78.

Decisions follow an absolute hierarchy: **STOP > AVOID > CAUTION > CONTINUE**.
Object description never competes with collision risk.

### Hysteresis

- A non-critical direction is held for at least `holdMs` (1400 ms).
- Changing it requires the new sector to beat the held one by
  `switchMarginPoints` (14 points).
- The new option must agree with itself for `confirmFrames` frames; a left↔right
  reversal requires one more.
- Returning to CONTINUE has its own shorter dwell, so the system stops nagging
  promptly without flickering.
- Emergencies bypass all of it.

When two sides score within six points of each other the tie is broken on
physical evidence — wider walkable gap first, then which side the widest opening
in the frame is actually on — rather than left by a coin flip that would
alternate.

### Confidence, and what it gates

Decision confidence blends the margin over going straight, the separation between
the two sides, perception quality and the primary obstacle's tracking stability.
Below `minDirectionConfidence` (0.52), or when frame quality is poor, the system
says "Obstacle ahead. Use caution." instead of naming a side. It does not invent
a direction it cannot justify.

## 8. Speech

Silence is the default. A clear path produces nothing at all.

Speech happens when the action changes, the scene signature changes (different
object or distance band), risk moves by at least 18 points, or a slow 5-second
reminder is due for a still-live instruction. Everything else is suppressed, and
the reason is visible on the diagnostics panel.

Phrases are instructions, never descriptions: "Move slightly left." not "Chair
ahead on your left, move slightly left." Object identity is on screen and
available on demand via *Describe scene*.

Emergencies bypass every cooldown, cancel the utterance in progress mid-word, and
clear the queue. They are subject only to a 2-second anti-stutter floor keyed on
*priority* rather than exact text — an earlier version keyed it on text, so a
hazard whose distance band wobbled alternated between two STOP phrasings and
bypassed the floor every frame.

A per-minute backstop limits routine chatter. Emergencies are exempt from it,
because letting them consume the budget would starve the user of the instruction
they need next.

"Path clear." is spoken once per obstruction episode, after the path has stayed
clear for `pathClear.stabilityMs` (1600 ms), and never when there was no
obstruction to clear.

## 9. Health, recovery and degradation

`VisionHealthMonitor` tracks camera FPS, inference FPS, latency percentiles,
dropped frames, model status, backend, worker status, track count, memory
warnings, last successful inference and consecutive failures.

`VisionWatchdog` is timer-driven rather than loop-driven on purpose: if the loop
itself has stopped — the failure that matters most — a loop-driven check would
never fire.

Recovery ladder, cheapest first, with backoff:

1. `RETRY` — the pipeline may just have hiccuped.
2. `RESTART_WORKER` — rebuild the worker and session in place.
3. `FALLBACK_BACKEND` — WebGPU → WASM SIMD → WASM → main-thread runtime.
4. `RELOAD_MODEL` — drop the IndexedDB entry and re-fetch, in case the cached
   bytes are corrupt.
5. `REINITIALISE` — full re-selection from the top of the plan.

A step "succeeding" means the step completed, not that results are flowing again;
the ladder resets to the bottom only on the next genuinely successful inference.
If it is exhausted, the system says *"Vision system unavailable. Please stop."*
and shows a Retry control. It does not keep pretending to guide.

Degradation ladder:

| Level | Capability | Behaviour change |
| --- | --- | --- |
| 1 | detector + tracking + refinement | Full directional guidance. |
| 2 | detector + tracking | Refinement disabled; directions still available. |
| 3 | detector only | Tracking untrustworthy, so **no directions** — warnings and stops only. |
| 4 | unreliable | Says so and asks the user to stop. |

Level 3 changes what the system is *willing to say*, not just what it computes. A
system that keeps issuing confident "move left" instructions from untrustworthy
data is more dangerous than one that admits it cannot see.

Degradation is immediate; restoration requires repeated confirmation.

## 10. Coordinate spaces

Four spaces, not interchangeable:

1. **Model** — pixels inside the square letterboxed input.
2. **Camera** — normalised over the raw video frame.
3. **Display** — normalised over the on-screen element after `object-fit: cover`
   and any CSS mirroring. Used for overlay drawing.
4. **Navigation** — normalised over the region the user can actually *see*,
   oriented so x=0 is their physical left. Used by every reasoning stage.

Navigation space is derived from the visible region deliberately. With
`object-fit: cover` a phone crops the sides of a 4:3 sensor frame in portrait; an
obstacle in that cropped strip is not in the user's field of view and must not
swing a direction decision. `cameraToNavigation()` also returns a `visibility`
fraction, which the risk engine uses to discount half-cropped objects.

The CSS in `css/style.css` and `CoordinateMapper`'s assumed fit must stay in
agreement: changing `object-fit` in one place requires changing it in the other.

## 11. Privacy

- No frame is recorded, stored or uploaded.
- No inference request leaves the device. The only network access is fetching
  model weights and the ONNX Runtime bundle, both cached afterwards.
- Once the model is cached, guidance works with no network at all.
- The camera stream is stopped on Stop and on `pagehide`.

## 12. Configuration

Tuning lives in `js/config/`:

| File | Contents |
| --- | --- |
| `visionConfig.js` | Detection thresholds, tracking gates, scheduler bounds, quality thresholds, watchdog timings, warm-up budget. |
| `navigationConfig.js` | Corridor geometry, free-space grid, proximity bands, risk weights, path scoring, hysteresis. |
| `speechConfig.js` | Priorities, cooldowns, phrase library, voice defaults. |
| `deviceProfiles.js` | Tier classification and per-tier/mode operating points. |
| `modelRegistry.js` | Model candidates with format, input size, quantisation and expected latency. |
| `classCatalog.js` | COCO order, hazard tiers, class priorities, ignore list. |

Values that were calibrated against measurements rather than chosen carry a
comment saying so — `quality.blurThreshold` and `RISK_NORMALISER` in particular.
Changing either will move test expectations, which is the intended safety net.
