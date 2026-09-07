# Changelog — Assistive Navigation System V2

All significant changes are recorded here.
Format: [Phase] Date — Description

---

## [Realtime] 2026-09-07 — Fix startup failure and add browser smoke testing

The client shipped in the previous entry did not start. Root causes, all found by
adding an on-page startup trace and then driving the real page in a real browser:

### Startup was fatally broken

- **`voiceManager.js` read `SpeechConfig.defaults.rate`.** That key was renamed to
  `voice` in the rewrite. Its constructor threw, and because it is constructed
  inside `SpeechManager` inside `NavigationApp`, the entire application failed to
  construct — blank screen, no status change, nothing on a phone to diagnose it
  with. Every module imported cleanly, so nothing caught it: the throw was at
  *construction*, not at import.
- **`speechQueue.js` read `SpeechConfig.Priority.DIRECTION_CHANGE`**, also renamed.
  That silently evaluated to `undefined`, and since every comparison against
  `undefined` is false, utterances appended instead of sorting by priority. The
  queue looked healthy and had quietly stopped prioritising.

### The detector could never be selected

- **The warm-up gate judged p95 across all passes, including the first.** With four
  samples, p95 *is* the maximum, which is always the cold pass that pays for shader
  compilation and kernel selection. Every backend was rejected for its one-time
  startup cost on every device. Now steady state (passes 2..n) is judged against
  the latency budget and the cold pass has its own, far larger budget.
- **The preallocated-output optimisation returned unwritten zero buffers.** Passing
  a pre-made tensor via ORT's `fetches` left it untouched, so every pass after the
  first produced all zeros. The sanity check caught it and rejected every ONNX
  candidate — and live detection would have been silently empty too. Removed;
  correctness beats saving a fixed 705 KB allocation.
- **An all-zero output is legitimate for an end-to-end head.** YOLO26n emits
  post-NMS detections, so zero detections is zero bytes of signal, not a fault.
  The sanity check is now head-aware.
- **Profile input size could contradict the model's static input shape.** A LOW-tier
  device, or FAST mode on any tier, asked for 256 while every exported ONNX graph
  is 320. The model now dictates the tensor size; the profile only ranks candidates.

### Hangs with no visible cause

- `navigator.gpu.requestAdapter()` was awaited unbounded. On some Android GPU
  drivers it never settles, leaving the app stuck on "Profiling device". Now
  bounded, as is every await on the startup path, with an overall ceiling.
- The ONNX Runtime import was unbounded and single-sourced. Now bounded per
  attempt, with a same-origin path tried first and two CDN mirrors after it.
- MediaPipe warmed up on `Date.now()` and ran live on `performance.now()`. The
  first live frame went backwards by a billion milliseconds, and MediaPipe rejects
  every call after a non-monotonic timestamp. The adapter now owns its clock.
- `OffscreenCanvas` was assumed to provide a 2D context. Safari shipped the
  constructor first, where `getContext("2d")` returns null and the pipeline would
  capture nothing, silently. The context is now verified with an element fallback.

### Behaviour that was too aggressive

- **Poor frame quality drove the system to level 4** (`UNRELIABLE`), which enters
  the error state and demands a manual retry. Pointing the camera at a blank wall
  should not require restarting the vision system. Quality now caps at level 3:
  directions are withheld, the user is told, and it recovers by itself.
- **Degradation flapped during the first second**, because the inference-rate check
  ran before the rolling FPS window had filled. Each downgrade spoke "Reduced
  accuracy." — three times in a ten-second run. The rate check now waits for a
  measurable sample.
- **The ground-freeness estimator read floor markings as obstacles.** Tiling grout,
  painted lines, expansion joints and shadow edges are thin bands with floor
  visible above them; an obstacle occludes everything above its base. A run-length
  requirement now distinguishes the two. Measured on a synthetic floor with
  scrolling seams: clearance 0.75 before, 1.0 after.
- The boot watchdog measured total elapsed time and fired mid-startup while a
  24-second WebGPU session build was legitimately running. It now measures lack of
  progress.

### Added

- `js/bootGuard.js` — a classic, ES5, zero-dependency script loaded before the
  module graph. It renders a startup trace on the page, captures uncaught errors,
  unhandled rejections and failed asset loads, watchdogs a stalled startup and
  offers a copyable report. A phone has no console; this is the console.
- `tools/smoke_browser.mjs` — drives the real page in a real Chromium with a
  canvas-backed synthetic camera and asserts that startup completes, a detector is
  selected and warmed up, inference produces results, no buffers leak, quality is
  usable, the path reads clear and no guidance is spoken on a clear path. Every
  bug above was found by this or by the trace it prints on failure.
- `tests/js/configContract.test.mjs` — verifies every config property the source
  reads actually exists, and that every subsystem can be **constructed**, not
  merely imported. That gap is what let the `defaults`/`voice` rename ship.
- `tests/js/startup.test.mjs` — pins the timeout utilities, capability probes,
  MediaPipe clock, boot-guard constraints and per-step tracing.

Suite is now 262 checks plus the browser smoke test. Verified on Chromium:
YOLO11n/YOLO26n on the ONNX Runtime WebGPU path, 13–16 inference FPS, p95
130–180 ms, no failures, no leaks.

---

## [Realtime] 2026-09-07 — Browser client re-engineered for stable walking guidance

Scope: the web client under `src/assistive_navigation/web/static/` only. The
Python desktop pipeline, its configuration and all published evaluation results
are untouched.

Audit of the previous client is in `docs/realtime_architecture_audit.md`.
New reference documentation: `docs/realtime_pipeline.md`, `docs/benchmarking.md`,
`docs/field_testing.md`.

### Root causes fixed

- **Inference blocked the UI.** The render loop awaited `detect()` inline. It now
  re-schedules `requestAnimationFrame` first and fires inference without awaiting
  it, with ONNX Runtime Web running in an ES-module Web Worker.
- **The full-resolution preview was the inference input.** Inference now runs on a
  dedicated letterboxed low-resolution stream from a reused `OffscreenCanvas`,
  with pooled tensor buffers transferred to the worker and transferred back.
  `VisionConfig.inputResolution` was previously declared but never referenced.
- **Confirmation gating was effectively disabled** (`temporalConfirmFrames: 1`,
  thresholds at 0.12–0.15). Thresholds raised to 0.30/0.38 with a genuine
  three-frame confirmation gate, plus a separate emergency channel so smoothing
  can never delay a STOP.
- **The barrier detector manufactured obstacles.** `barrierDetector.js` declared a
  frame-filling "wall" at 0.85 confidence whenever the image centre was
  low-texture, which fired on plain floors, roads and sky and turned each one into
  a false STOP. Removed and replaced with a ground-freeness estimator that finds
  where the walkable floor actually ends.
- **Failures were silent and permanent.** `detectFrame` caught every error and
  returned the previous frame's detections forever. Added a health monitor,
  timer-driven watchdog, five-step recovery ladder and four-level degradation
  ladder. The system now either works, recovers, or says so.
- **Hysteresis did not hold.** The old implementation only blocked a strictly
  opposite direction and reset its own hold counter on every decision. Replaced
  with dwell time, a 14-point switch margin, multi-frame confirmation and a
  deterministic, evidence-based tie-break.
- **Direction ignored whether the target side was safe.** Path scoring now scores
  left, centre and right independently from free space and per-sector risk, and
  a side containing a high-risk obstacle is never eligible.
- **Free space was a bounding-box tally.** Replaced with a 24-column occupancy grid
  carrying per-column free depth, ground-band weighting, shoulder-width passability
  and optional refinement fused with `min` semantics.
- **Speech was long and globally rate-limited.** Phrases are now short
  instructions; suppression is keyed on action, scene signature and material risk
  change; emergencies bypass cooldowns and interrupt mid-utterance; "Path clear."
  is spoken once per obstruction episode.
- **No model caching or warm-up.** Model bytes are cached in IndexedDB and warm-up
  passes run before the user is told the system is ready.
- **No coordinate mapping.** Added an explicit four-space mapper (model, camera,
  display, navigation) that accounts for letterboxing, `object-fit` cropping,
  rotation and mirroring, and reports per-object visibility after the crop.

### Bugs found by the new tests

- **NMS suppressed almost everything.** Candidate boxes were read through a shared
  scratch array, so comparing box *i* against box *j* overwrote box *i*: every IoU
  evaluated to 1 and every detection after the first of each class was silently
  dropped. Found by cross-checking the browser decoder against an independent
  NumPy postprocess on real ONNX output (`tests/js/decodeParity.test.mjs`).
- **The corridor fed back into its own inputs.** Steering the corridor centre
  towards the widest free-space gap shrank the sector on that side and inverted
  the path scores, producing the left/right oscillation the upgrade was meant to
  remove. The centre no longer moves; choosing a side is the decision engine's job.
- **Out-of-bounds typed-array reuse produced NaN.** The ground-freeness estimator
  reused a column-length `Int32Array` as pixel-width scratch; out-of-range writes
  were silently dropped and reads returned `undefined`.
- **`blurThreshold` was off by an order of magnitude** (0.020 against measured
  values of ~0.0007 for a smeared frame and ~0.007 for a structured one), so every
  frame read as motion-blurred. Recalibrated to 0.0025 against measurements.
- **The speech rate window grew for the whole session**, because emergency and
  path-status utterances reached `commit()` without passing the trimming branch.
- **The emergency anti-stutter floor was keyed on exact text**, so a hazard whose
  distance band wobbled alternated between two STOP phrasings and bypassed the
  floor every frame — 62 utterances per minute in the simulated walk. Now keyed on
  priority; measured at 22 per minute on a sequence that walks into an obstacle
  every ten seconds.

### Added

- Model export: `scripts/export_web_models.py` produces YOLO11n and YOLO26n ONNX
  at 320 with uint8 variants plus a manifest, self-hosted so the client has no
  third-party model CDN dependency and works offline once cached.
- Benchmark screen: measures each candidate on the actual device across six
  operator-guided conditions and reports inference FPS, average and p95 latency,
  heap, detections per frame, dropped frames and camera FPS with a verdict.
- Diagnostics dashboard covering the detector, timing, perception, decision and
  system state.
- Demo mode with a start-up gate and the ten-station test course with observer
  checklist.
- Scenario replay running the same scripted situations as the automated suite,
  on-device, with no camera involved.
- Automated suite: 224 checks via `node --test`, covering decoding, coordinate
  transforms, tracking, risk, path scoring, decisions, hysteresis, speech policy,
  scheduling, quality, scene change, backend selection, degradation, recovery,
  long-run stability and static integrity of the un-bundled client.
- Decoder parity fixtures generated from real ONNX output
  (`tools/make_decode_fixtures.py`).

### Removed

`vision/visionDetector.js`, `vision/temporalTracker.js`,
`vision/barrierDetector.js`, `vision/detectionTypes.js`,
`navigation/walkingCorridor.js`, `navigation/spatialAnalysis.js`,
`navigation/sceneNarrative.js`, `navigation/stateMachine.js`,
`ui/audioFirstUI.js`, `audio/speechEngine.js`, `camera/orientation.js`,
`utils/geometry.js`, `utils/smoothing.js`, `utils/timing.js`.

Retained: the camera constraint fallback ladder, `audio/voiceManager.js`,
`audio/speechQueue.js`, and the TensorFlow.js COCO-SSD path as the guaranteed-load
fallback detector.

### Not yet verified

On-device confirmation is outstanding. The machinery for camera responsiveness,
model caching, backend fallback, worker recovery, rotation handling, thermal
behaviour, the five- and ten-minute walking tests and cross-device degradation is
implemented and unit-tested, but requires a physical smartphone. See
`docs/field_testing.md` §9 for the item-by-item status.

---

## [Packaging] 2026-09-04 — Fix `python -m assistive_navigation` startup

Isolated packaging/installation follow-up on top of the published Phase 0–18
baseline. No pipeline behaviour, configuration values, or evaluation results
changed.

- Fixed the invalid build backend in `pyproject.toml`
  (`setuptools.backends.legacy:build` → `setuptools.build_meta`). The old value
  does not exist, which is why `pip install -e .` failed and
  `python -m assistive_navigation` reported `No module named ...`.
- Added `src/assistive_navigation/__main__.py` (5 lines) delegating to
  `main.main()` so the package is runnable via `python -m assistive_navigation`.
  No pipeline logic added.
- Installed the project editable into the venv (`pip install -e .`).
- Verified: full non-hardware regression **570 passed, 0 failed**;
  `python -m assistive_navigation --help` and `--headless` resolve correctly.
- Updated startup docs (README, `docs/installation.md`, `demo/demo_script.md`)
  to include the one-time `pip install -e .` step.

---

## [Phase 18] 2026-09-04 — Final Demonstration & Validation

Validation, evidence, and presentation only. No pipeline source,
`config/config.yaml`, tests, evaluation harnesses, or Phase 4/15/16 artifacts
were modified (verified byte-identical via SHA256 + git tree hashes).

- Software verification: full non-hardware regression **570 passed, 0 failed**;
  protected paths byte-identical to the Phase 17 baseline.
- Added `demo/` evidence package: `test_output.txt`, `git_status.txt`,
  `git_log.txt`, `protected_file_hashes.txt`, `project_tree.txt`, `README.md`
  (evidence index), and `demo_script.md` (operator runbook for the live
  demonstration).
- Added `docs/final_report.md` (problem → future work; every quantitative claim
  labelled [MEASURED]/[SYNTHETIC-DETERMINISTIC]/[OPERATOR-LIVE]/[NOT RUN]/
  [DESIGN TARGET]/[UNVALIDATED ASSUMPTION]; no precision/recall/mAP/accuracy/
  generalization claims).
- Added `docs/presentation.md` (~14-slide outline; only supported results).
- Updated `README.md`, `PROJECT_STATUS.md` for Phase 18 completion.
- Live webcam demonstration, live audio verification, live empty-scene FP
  measurement, and live performance runs are marked **[NOT RUN]** — they
  require physical execution and were NOT fabricated.

---

## [Phase 17] 2026-09-04 — Documentation & Reproducibility

Documentation and dependency-manifest correction only. No pipeline source,
`config/config.yaml`, tests, evaluation harnesses, or Phase 4/15/16 artifacts
were modified.

- Corrected `requirements.txt` to match the verified installed environment:
  added `ultralytics==8.4.137`, `torch==2.13.0`, `torchvision==0.28.0` (exact
  installed versions; no upgrades/downgrades). Documented that `scipy` and
  `piper-tts-plus` are NOT installed, and the INT8 depth-model rejection.
- Rewrote `README.md` for the current state (Phase 16 complete): corrected the
  phase table and technology stack (YOLO11n via Ultralytics/PyTorch `.pt`;
  Depth Anything V2 Small via ONNX Runtime), added the five-label evidence key,
  citation/attribution, limitations, and future work; preserved the safety
  disclaimer and privacy/local-processing sections.
- Filled five documentation stubs: `docs/architecture.md` (architecture +
  data flow + per-stage explanations, items 4–18), `docs/installation.md`
  (clean-machine setup, requirements, model acquisition, configuration),
  `docs/models.md` (model sources/sizes/licenses + INT8-rejection rationale),
  `docs/testing.md` (layout, markers, counts), `docs/troubleshooting.md`.
- Added `docs/usage.md` (run/debug/headless/tests/evaluation/hardware) and
  `docs/reproducibility.md` (reproduce Phase 4/15/16 + reproducibility,
  clean-machine, final-evaluation, and final-demo checklists).
- Updated `docs/licenses.md`: added torch + torchvision (BSD-3-Clause) with
  installed versions; corrected scipy and piper-tts-plus to NOT-installed;
  refreshed the "Last updated" line.
- `docs/evaluation.md` preserved verbatim (Phase 4 + Phase 16 content).
- All quantitative claims labelled [MEASURED] / [SYNTHETIC-DETERMINISTIC] /
  [OPERATOR-LIVE / NOT RUN] / [DESIGN TARGET] / [UNVALIDATED ASSUMPTION].
  No precision/recall/mAP/accuracy/generalization claimed.
- Full non-hardware regression: **570 tests pass**. Protected baselines
  (`config.yaml`, `phase4_results.csv`, `phase4_report.txt`) verified
  byte-identical (SHA256) before and after.

---

## [Phase 16] 2026-09-04 — Final System Evaluation

- Added `tests/evaluation/phase16_pipeline_eval.py`: Layer 2 synthetic,
  deterministic pipeline evaluation using the real stage classes with
  constructed ground truth. 73 checks (48 exact + 7 boundary + 18 heuristic),
  all pass. Covers direction (≥21 positions + boundaries), proximity bands +
  monotonicity, temporal confirmation, area gate @0.15, tracking
  persistence/coasting, alert correctness/triggers/text/selection/dedup/count,
  pipeline-internal alert latency (excludes TTS/audio), multi-object,
  partial small-area, and the difficult background-FP profile.
- Added `tools/phase16_final_eval.py`: Layer 1 detection-level roll-up
  (reuses `phase4_results.csv` read-only; reproduces the Phase 15 replay
  exactly) + Layer 3 hardware capture/analysis (copies `performance.csv` to
  timestamped `phase16_perf_<ts>.csv`; FPS mean/median/min/max; detector/depth/
  total latency; depth interval; system-wide CPU; process RSS; live FP) +
  timestamped `phase16_report_<ts>.txt` generator with explicit NOT RUN markers.
- Added `tests/unit/test_phase16.py`: 35 deterministic tests (Layer 2 behavior,
  Layer 1 replay reproduction, Layer 3 perf-math on synthetic CSV, and
  baseline/config guards).
- Appended a Phase 16 section to `docs/evaluation.md` (Phase 4 section preserved
  verbatim).
- Full regression: **570 tests pass** (535 prior + 35 new).
- No thresholds, detector/depth/tracker/TTS architecture, config, or Phase 4/15
  baselines were changed. `phase4_results.csv`, `phase4_report.txt`, and
  `config.yaml` verified byte-identical (SHA256) before/after.
- Layer 3 hardware evaluation was **NOT RUN** in this pass (no live camera
  executed); no hardware results were fabricated.

---

## [Phase 15] 2026-09-03 — False-Positive Reduction

- Added `tools/phase15_fp_analysis.py`: offline, READ-ONLY analysis of the
  existing Phase 4 evaluation CSV. Recomputes `area_norm` from stored
  640×480 bboxes, separates genuine vs false-positive accepted detections,
  and sweeps candidate `min_area_norm` thresholds with before/after
  suppression + retention numbers. Writes timestamped, separate output
  (`data/evaluation/phase15_analysis_<ts>.{txt,csv}`).
- Calibrated `config/config.yaml` `temporal.min_area_norm`: **0.02 → 0.15**
  (single documented value, evidence-based; not forced).
  - FP suppression: 72/303 (23.8%) → 215/303 (**71.0%**)
  - Genuine retention: 114/114 (100%) preserved — zero genuine loss
  - Dominant empty-scene background `person` FP: 203 → 22 accepted
- Left `confidence_threshold` at 0.35 and `confirmation_frames` at 8
  (unchanged). No change to detector, tracker, depth model, fusion,
  spatial, priority/temporal logic, TTS, audio queue, or alert manager.
- Did NOT modify `phase4_results.csv` or `phase4_report.txt`.
- Added 38 tests: `tests/unit/test_phase15.py` (area math, CSV
  classification, gate behaviour, sweep reproducibility, recommendation
  safety, live-config checks) and `tests/evaluation/phase15_fp_replay.py`
  (auditable before/after replay of the raw CSV).
- Full regression: 535 logic + integration tests pass.
- Documented limitations NOT addressed by the area gate: class-confusion
  mislabels (FP-3), door/stair COCO gap (FP-4), and the ~29% of empty-scene
  FP whose area overlaps the genuine cluster. `couch_visible` still NOT TESTED.

---

## [Phase 1] 2026-09-01 — Environment & Repository Setup

- Initialized Git repository
- Created Python 3.10.11 isolated virtual environment (.venv)
- Created full project folder structure
- Created .gitignore (excludes .venv, models, logs, .env, cache files)
- Created .env.example template
- Created config/config.yaml with all configurable parameters documented
- Created README.md with safety disclaimer
- Created PROJECT_STATUS.md
- Created CHANGELOG.md
- Created pyproject.toml and requirements.txt
- Created MIT LICENSE
- Created all Python skeleton module files (__init__.py + stub files)
- Installed Phase 1 packages: opencv-python, onnxruntime, pyttsx3, psutil, pyyaml, pytest
- Verified: .venv activates, pytest runs, config loads
- Initial Git commit created

---

## [Phase 0] 2026-09-01 — Discovery & Planning

- System discovery completed
- Architecture designed
- Technology options compared and selected
- All licenses verified
- Phase 0 report produced and approved
