# Project Status — Assistive Navigation System V2

---

## Current Phase

**Phase 15 — False-Positive Reduction** ✅ COMPLETE
**Next Phase: Phase 16 — Final Evaluation** ⏳ Awaiting approval

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
