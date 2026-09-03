# Changelog — Assistive Navigation System V2

All significant changes are recorded here.
Format: [Phase] Date — Description

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
