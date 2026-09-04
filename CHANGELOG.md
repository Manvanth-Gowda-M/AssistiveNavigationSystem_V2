# Changelog — Assistive Navigation System V2

All significant changes are recorded here.
Format: [Phase] Date — Description

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
