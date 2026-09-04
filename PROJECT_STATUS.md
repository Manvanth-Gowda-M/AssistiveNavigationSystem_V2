# Project Status — Assistive Navigation System V2

---

## Current Phase

**Phase 17 — Documentation & Reproducibility** ✅ COMPLETE
**Next Phase: Phase 18 — Final Demonstration** ⏳ Awaiting approval

---

### Phase 17 — Documentation & Reproducibility ✅ COMPLETE (2026-09-04)

Documentation and manifest correction only. No pipeline source, config, tests,
evaluation harnesses, or Phase 4/15/16 artifacts were modified.

- [x] `requirements.txt` corrected to match the verified installed `.venv`:
      added `ultralytics==8.4.137`, `torch==2.13.0`, `torchvision==0.28.0`
      (exact installed versions — no upgrades/downgrades); documented that
      scipy and piper-tts-plus are NOT installed; documented the INT8 depth
      rejection.
- [x] `README.md` rewritten for the current state: Phase 16 complete, corrected
      phase table and technology stack (YOLO11n via Ultralytics/PyTorch `.pt`;
      depth via ONNX Runtime), five-label evidence key, preserved safety
      disclaimer and privacy sections, added citation/attribution, limitations,
      and future work.
- [x] Filled the five documentation stubs: `docs/architecture.md`,
      `docs/installation.md`, `docs/models.md`, `docs/testing.md`,
      `docs/troubleshooting.md`.
- [x] Created `docs/usage.md` and `docs/reproducibility.md` (with the
      reproducibility, clean-machine, final-evaluation, and final-demo checklists).
- [x] `docs/licenses.md`: added torch + torchvision (BSD-3-Clause) with
      installed versions; corrected scipy and piper-tts-plus to NOT-installed;
      updated the "Last updated" line.
- [x] `docs/evaluation.md` left unchanged (Phase 4 + Phase 16 content preserved
      verbatim; referenced from the new docs).
- [x] Full non-hardware regression re-verified: 570 pass. Protected files
      (config.yaml, phase4_results.csv, phase4_report.txt) verified byte-identical.
- [x] All quantitative claims tagged with the five evidence labels; no
      precision/recall/mAP/accuracy/generalization claimed.

---

### Phase 16 — Final System Evaluation ✅ COMPLETE (2026-09-04)

Rigorous three-layer evaluation of the complete system. Evaluation code and
documentation only — no thresholds, pipeline modules, config, or Phase 4/15
baselines were changed.

- [x] **Layer 1 — Detection-level** (offline, `phase4_results.csv` read-only):
      reproduces Phase 15 replay exactly (0.02→0.15: 72→**215/303** FP
      suppressed, **114/114 genuine retained, 0 lost**); reports per-class
      retention, accepted FP by scene, 210 wrong-class observations (FP-3),
      and scene-specific no-detection counts (NOT a generalized FN rate).
- [x] **Layer 2 — Synthetic pipeline** (`tests/evaluation/phase16_pipeline_eval.py`):
      73 deterministic checks (48 exact + 7 boundary + 18 heuristic), all pass.
      Direction (≥21 positions + boundaries), proximity bands + monotonicity,
      temporal confirmation, area gate @0.15, tracking persistence/coasting,
      alert correctness/triggers/text/selection/dedup/count, pipeline-internal
      alert latency (excludes TTS/audio), multi-object, partial small-area,
      and the difficult background-FP profile (never confirms, no alert).
- [x] **Layer 3 — Hardware/live** (`tools/phase16_final_eval.py`): capture +
      analysis mechanism (copies `performance.csv` to timestamped
      `phase16_perf_<ts>.csv`; FPS mean/median/min/max; detector/depth/total
      latency; depth inference count + interval; system-wide CPU; process RSS;
      live empty-scene FP). **NOT RUN in this pass** — no hardware executed;
      no results fabricated.
- [x] Tests: 35 new (`tests/unit/test_phase16.py`); full regression **570 pass**.
- [x] Baselines verified unchanged (SHA256 identical before/after):
      `phase4_results.csv`, `phase4_report.txt`, `config.yaml`.
- [x] `min_area_norm=0.15`, `confidence_threshold=0.35`, `confirmation_frames=8`
      all unchanged.

Documented limitations carried forward: FP-3 mislabels, door/stairs COCO gap,
remaining large-area FP overlap, `couch_visible` NOT TESTED, single environment,
relative depth, synthetic ground truth is by construction, alert latency
excludes TTS.

---

### Phase 15 — False-Positive Reduction ✅ COMPLETE (2026-09-03)

Goal: reduce the dominant Phase 4 false positives WITHOUT changing the
detector, tracker, depth model, TTS, or confidence threshold, and with a
measurable before/after justification.

- [x] Offline area analysis tool (`tools/phase15_fp_analysis.py`) — reads
      the existing `phase4_results.csv` READ-ONLY, recomputes `area_norm`
      from the stored 640×480 bboxes, separates genuine vs false-positive
      accepted detections, and sweeps candidate `min_area_norm` thresholds.
- [x] Evidence (pre-temporal, DetectionFilter output only):
      - GENUINE accepted n=114 → area_norm min 0.1612, median 0.5156
      - FALSE-POSITIVE accepted n=303 → area_norm min 0.0025, median 0.0645
- [x] Calibrated `temporal.min_area_norm` 0.02 → **0.15** (single value).
      - Before (0.02): 72/303 FP suppressed (23.8%), 114/114 genuine kept
      - After (0.15): 215/303 FP suppressed (**71.0%**), 114/114 genuine kept
      - +143 additional FP suppressed at **zero genuine-object loss**
      - 0.20 rejected: genuine retention would drop to 82.5% (20 lost)
- [x] Dominant FP-1 (empty-scene background `person`): 203 → 22 accepted
      (181 suppressed, ~89%)
- [x] Per-class genuine retention @ 0.15: person 30/30, chair 26/26,
      laptop 29/29, bottle 29/29
- [x] confidence_threshold UNCHANGED (0.35); confirmation_frames UNCHANGED (8)
- [x] Detector, tracker, depth (Depth Anything V2 Small), fusion, spatial,
      priority, temporal logic, TTS, audio queue, alert manager: UNTOUCHED
- [x] `phase4_results.csv` / `phase4_report.txt`: NOT modified
- [x] Tests: 38 new Phase 15 tests (`tests/unit/test_phase15.py`,
      `tests/evaluation/phase15_fp_replay.py`), all pass
- [x] Full regression: 535 logic + integration tests pass (0 failures)

Documented limitations NOT solved by the area gate (honest):
  - FP-3: class-confusion mislabels on real objects (table→bottle etc.)
  - FP-4: doors/stairs — YOLO11n COCO capability gap
  - ~29% of empty-scene-family FP overlap the genuine area cluster and are
    intentionally NOT suppressed (raising the threshold would drop genuine
    objects). `couch_visible` remains NOT TESTED.

---

### Phase 2 — Camera Capture Module ✅ COMPLETE
- [x] capture.py implemented (CameraCapture class, CameraError exception)
- [x] Context manager (with statement) support — guarantees release
- [x] Synchronous read() with FPS cap enforcement
- [x] measure_capture_fps() benchmark method
- [x] get_smoothed_fps() for live display (Phase 13)
- [x] Graceful failure on invalid device (CameraError, not crash)
- [x] All settings read from config.yaml (no magic numbers)
- [x] 30/30 tests pass in test_camera.py
- [x] Actual measured FPS: 21.36 FPS (minimum: 15.0) — PASS
- [x] Git commit made

### Phase 0 — Discovery & Planning ✅
- System discovery completed (OS, CPU, GPU, Python, webcam, audio)
- Full architecture designed (18-stage modular pipeline)
- Technology options researched and compared
- All licenses verified
- Cost plan confirmed ($0)
- Recommended stack selected:
  - Detection: YOLO11n via ONNX Runtime
  - Tracking: ByteTrack (Ultralytics built-in)
  - Depth: Depth Anything V2 Small (Apache-2.0)
  - TTS: pyttsx3 + Windows SAPI5
- Phase 0 report approved

### Phase 1 — Environment & Repository Setup ✅ COMPLETE
- [x] Workspace confirmed empty
- [x] `.venv` created (Python 3.10.11, isolated)
- [x] Git repository initialized
- [x] Project folder structure created (20 directories)
- [x] `.gitignore` created
- [x] `.env.example` created
- [x] `config/config.yaml` created (all parameters documented)
- [x] `README.md` created (with safety disclaimer)
- [x] `PROJECT_STATUS.md` created
- [x] `CHANGELOG.md` created
- [x] `pyproject.toml` created
- [x] `requirements.txt` created and pinned
- [x] `LICENSE` created (MIT)
- [x] All Python skeleton files created (69 files total)
- [x] Phase 1 packages installed: opencv-python==5.0.0.93, onnxruntime==1.23.2, pyttsx3==2.99, psutil==7.2.2, PyYAML==6.0.3, pytest==9.1.1
- [x] config_loader.py fully implemented and tested
- [x] 8/8 config tests pass
- [x] Initial Git commit made (d2aaf1a)
- [x] All Phase 1 PASS conditions verified

---

## Current Tests

None yet — Phase 1 is not complete.

---

## Known Issues

None yet.

---

## Key Hardware Facts

- **CPU only** — Intel i5-12450H, 8 cores, no CUDA GPU
- **Python** — 3.10.11 in isolated .venv
- **Webcam** — Integrated camera confirmed working
- **Audio** — Realtek HD + SAPI5 voices (David, Zira) confirmed

---

## Next Step

**Phase 2 — Camera Capture Module** — awaiting your approval to proceed.

Phase 2 will:
1. Implement `src/assistive_navigation/camera/capture.py`
2. Test that the webcam opens, captures frames, measures FPS, and closes cleanly
3. Run `tests/unit/test_camera.py`
4. Verify all Phase 2 PASS conditions before proceeding to Phase 3
