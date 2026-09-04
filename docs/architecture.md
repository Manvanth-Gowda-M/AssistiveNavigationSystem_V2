# Architecture & Data Flow

This document explains the complete system architecture, the end-to-end data
flow, and every pipeline stage. It covers README documentation items 4–18.

Quantitative claims use the project's five evidence labels:
**[MEASURED]**, **[SYNTHETIC-DETERMINISTIC]**, **[OPERATOR-LIVE / NOT RUN]**,
**[DESIGN TARGET]**, **[UNVALIDATED ASSUMPTION]**. Detailed evaluation numbers
live in [`evaluation.md`](evaluation.md); this file links rather than
duplicates them.

---

## 4. Complete architecture

The system is a single-process, CPU-only pipeline orchestrated by
`AssistiveNavigationPipeline` in `src/assistive_navigation/main.py`. Each
stage is an independent module with its own dataclass output. Stages are
composed in a fixed order and isolated by per-stage exception handling so a
failure in one stage degrades gracefully rather than crashing the loop.

```
                         config/config.yaml
                                 │  (single source of truth for all thresholds)
                                 ▼
 ┌───────────┐   frame   ┌────────────┐  DetectionResult  ┌───────────────┐
 │  Camera   │──────────▶│  Detector  │──────────────────▶│    Filter     │
 │ (OpenCV)  │           │ YOLO11n/PT │                   │ conf + classes│
 └───────────┘           └────────────┘                   └──────┬────────┘
                                                    accepted List[Detection]
                                                                  ▼
 ┌───────────────┐        List[TrackedObject]        ┌────────────────────┐
 │ Depth (ONNX)  │  depth_map  ┌──────────────────┐  │  Tracker (ByteTrack)│
 │ every N frames│────────────▶│   DepthFusion    │◀─┤  stable track_ids   │
 └───────────────┘             │  depth+priority  │  └────────────────────┘
                               └───────┬──────────┘
                             List[FusedObject]
                                       ▼
 ┌──────────────┐  List[DirectedObject]  ┌───────────────────┐
 │SpatialReasoner│──────────────────────▶│NavigationPriority │
 │ L / C / R     │                        │  nav_score        │
 └──────────────┘                         └────────┬──────────┘
                                       List[ScoredObject]
                                                   ▼
 ┌──────────────────────┐  List[ConfirmedObject]  ┌───────────────┐
 │TemporalConfirmation  │────────────────────────▶│ AlertManager  │
 │ 5 gates, per-track   │                          │ 1 alert/cycle │
 └──────────────────────┘                          └──────┬────────┘
                                                    alert text
                                                           ▼
                                        ┌──────────────┐   ┌──────────┐
                                        │ SimpleAudio  │──▶│  TTS     │
                                        │   Queue      │   │ pyttsx3  │
                                        └──────────────┘   └──────────┘

  PerfMonitor wraps every frame (timing, CPU system-wide, RSS, CSV log).
  DebugView optionally renders an overlay window (skipped when headless).
```

Design principles:
- **One config file** (`config/config.yaml`) holds every threshold. No magic
  numbers in code.
- **Progressive enrichment**: each stage adds fields to a dataclass and passes
  the rest through unchanged (`Detection` → `TrackedObject` → `FusedObject` →
  `DirectedObject` → `ScoredObject` → `ConfirmedObject`).
- **Graceful degradation**: depth, audio, and debug are non-critical; only the
  camera and detector are critical to start.

---

## 5. End-to-end data flow (one frame)

`main.run_frame()` executes these steps in order every frame:

1. **Camera** → BGR frame (or `None` → frame skipped).
2. **Detector** → `DetectionResult` (raw detections + latency).
3. **Filter** → `FilterResult` (`accepted` list + `rejected` with reasons).
4. **Tracker** → `List[TrackedObject]` (stable `track_id`s).
5. **Depth** → depth map, but **only every Nth frame** (`depth_update_interval`,
   default 5); between updates a cached map is reused and `depth_age` increments.
6. **Fusion** → `List[FusedObject]` (adds relative depth + proximity + priority).
7. **Spatial** → `List[DirectedObject]` (adds LEFT/CENTER/RIGHT).
8. **Priority** → `List[ScoredObject]` (adds `nav_score`, sorted).
9. **Temporal** → `List[ConfirmedObject]` (adds confirmation state).
10. **AlertManager** → at most one `AlertResult` per frame → enqueued to audio.
11. **PerfMonitor.record/end_frame** → per-frame timing + resources → CSV.

`FrameResult` summarizes each frame (counts, latencies, `pipeline_fps`) for
logging and tests.

---

## 6–16. Pipeline stage explanations

### 6/7. Object detection and filtering
- **Detector** (`detection/detector.py`): YOLO11n via the **Ultralytics API on
  PyTorch** (`.pt` weights; the config stores an `.onnx` path but the detector
  overrides the extension to `.pt` and loads via Ultralytics). Returns raw
  detections above a low internal threshold.
- **Filter** (`detection/filter.py`): a three-layer model — RAW (everything) →
  CONFIDENT (`confidence_threshold`, default **0.35**) → FILTERED (only classes
  in the navigation lists; `ignored` classes and unknown classes are rejected
  **with a recorded reason**, never silently dropped).
- Behavior characterized physically in Phase 4. [MEASURED — see `evaluation.md`]

### 8. ByteTrack tracking
- **Tracker** (`tracking/tracker.py`): wraps Ultralytics' ByteTrack directly
  (no ReID model — lighter for CPU). Assigns stable `track_id`s, ages tracks,
  and coasts through brief detection gaps (`max_missed_frames`). Adds
  `age`, `last_seen`, `stability`, `area_norm`.
- Single-ID persistence and short-occlusion coasting validated in Phase 16
  Layer 2. [SYNTHETIC-DETERMINISTIC]

### 9. Depth Anything V2 Small
- **DepthEstimator** (`depth/depth_estimator.py`): the **Small (float32) ONNX**
  model via **ONNX Runtime** (CPU). The Small variant is Apache-2.0. Runs every
  `depth_update_interval` frames to control CPU cost; the cached map is reused
  between runs. The INT8 variant was **rejected** in Phase 6 (onnxruntime lacks
  its `ConvInteger` operator) — see [`models.md`](models.md).
- Depth-interval adherence is asserted in the Phase 14 tests. [MEASURED]

### 10. Relative proximity interpretation
- **DepthFusion** (`depth/fusion.py`) samples the central ROI of each track's
  bbox from the depth map, applies a trimmed median + per-track temporal
  smoothing, and classifies into FAR / MEDIUM / CLOSE / VERY_CLOSE using
  configured thresholds (very_close 0.75, close 0.55, medium 0.35).
- **Critical:** these are **relative** values (0–1, higher = closer within the
  current scene), **not metric distances**. [UNVALIDATED ASSUMPTION]
- Band membership + monotonicity validated in Phase 16 Layer 2. [SYNTHETIC-DETERMINISTIC]

### 11. Spatial direction
- **SpatialReasoner** (`navigation/spatial.py`): classifies each object's
  normalized `center_x` into LEFT / CENTER / RIGHT using half-open intervals
  (LEFT `[0, 0.35)`, CENTER `[0.35, 0.65)`, RIGHT `[0.65, 1.0]`). Out-of-range
  values are clamped; non-finite → UNKNOWN.
- Direction accuracy validated over ≥21 positions + boundaries in Phase 16
  Layer 2. [SYNTHETIC-DETERMINISTIC]

### 12. Navigation priority
- **NavigationPriorityEngine** (`navigation/priority.py`): computes a
  `nav_score` in [0,1] as a weighted sum of proximity, object importance,
  direction, confidence, and stability (weights in config). Sorts objects so
  the most urgent is first.
- Weights are **starting estimates**. [DESIGN TARGET]

### 13. Temporal confirmation
- **TemporalConfirmationFilter** (`navigation/temporal.py`): per-track state
  machine (NEW → CANDIDATE → CONFIRMED, plus LOST) with five gates: minimum
  bbox area (`min_area_norm`, **0.15** after Phase 15), consecutive frame count
  (`confirmation_frames`, default 8), class stability, position stability, and a
  minimum nav_score floor. Only CONFIRMED objects are eligible for alerts.
- State-machine behavior + the 0.15 area gate validated in Phase 16 Layer 2.
  [SYNTHETIC-DETERMINISTIC]

### 14. Alert manager
- **AlertManager** (`alerts/alert_manager.py`): consumes confirmed objects and
  issues **at most one alert per cycle**. Eligibility requires
  `is_confirmed AND direction != UNKNOWN AND nav_score > 0`. Trigger reasons (in
  order): `new_object`, `proximity_escalation`, `direction_change`,
  `cooldown_expired`. Selects the highest-`nav_score` candidate; formats a short
  phrase from config templates; suppresses duplicates within `cooldown_seconds`
  (default 5.0s). Timing thresholds are **starting estimates**. [UNVALIDATED ASSUMPTION]
- Trigger/selection/dedup/latency behavior validated in Phase 16 Layer 2.
  Alert latency there is **pipeline-internal** and **excludes** TTS/audio
  playback. [SYNTHETIC-DETERMINISTIC]

### 15. TTS / audio queue
- **SimpleAudioQueue** (`audio/queue.py`) buffers alert text; **TextToSpeech**
  (`audio/tts.py`) speaks via pyttsx3 + Windows SAPI5 in a background thread so
  the detection loop never blocks. Audio is non-critical: if TTS is unavailable,
  the pipeline continues silently.

### 16. Performance monitoring
- **PerfMonitor** (`utils/perf_monitor.py`): per-frame timing (total, detector,
  depth), rolling FPS, and — via psutil — **system-wide** CPU (`cpu_percent_system`)
  and **process** RSS (`rss_mb_process`), written to `logs/performance.csv`
  (overwritten each run). CPU is explicitly labelled system-wide to avoid
  implying per-process measurement.

---

## 17. False-positive reduction (Phase 15)

Phase 15 calibrated a single parameter — `temporal.min_area_norm` 0.02 → **0.15**
— using an offline analysis of the Phase 4 data. It suppresses the dominant
empty-scene background false positives while retaining all measured genuine
objects. Full evidence, before/after numbers, and limitations:
[`evaluation.md`](evaluation.md) (Phase 15 sections). [MEASURED]

## 18. Three-layer evaluation (Phase 16)

Phase 16 evaluates the whole system in three clearly separated layers:
- **Layer 1 — detection-level** (offline, reuses Phase 4 data). [MEASURED]
- **Layer 2 — synthetic pipeline** (deterministic, by construction). [SYNTHETIC-DETERMINISTIC]
- **Layer 3 — hardware/live** (operator-run; mechanism exists). [OPERATOR-LIVE / NOT RUN]

Methodology, checks, and PASS/FAIL gates: [`evaluation.md`](evaluation.md)
(Phase 16 section). No precision/recall/mAP/accuracy/generalization is claimed.

---

## Source module map

| Module | Responsibility |
|---|---|
| `main.py` | Pipeline orchestration, `run_frame()`, startup/shutdown |
| `camera/capture.py` | Webcam capture, FPS measurement |
| `detection/detector.py` | YOLO11n inference (Ultralytics/PyTorch) |
| `detection/filter.py` | Confidence + class filtering with reasons |
| `tracking/tracker.py` | ByteTrack multi-object tracking |
| `depth/depth_estimator.py` | Depth Anything V2 Small (ONNX) + interval scheduling |
| `depth/fusion.py` | ROI depth sampling, smoothing, proximity, priority |
| `navigation/spatial.py` | LEFT/CENTER/RIGHT direction |
| `navigation/priority.py` | `nav_score` computation + ranking |
| `navigation/temporal.py` | Temporal confirmation state machine |
| `alerts/alert_manager.py` | Alert decision, dedup, formatting |
| `audio/queue.py`, `audio/tts.py` | Non-blocking speech output |
| `utils/perf_monitor.py` | Timing + CPU/RAM metrics + CSV |
| `utils/config_loader.py`, `utils/logger.py` | Config + logging |
| `visualization/debug_view.py` | Optional OpenCV overlay |
