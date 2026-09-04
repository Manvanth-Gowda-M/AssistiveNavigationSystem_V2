# Testing

Covers README item 25. Explains the test layout, markers, how to run subsets,
and the current counts.

---

## Test layout

```
tests/
├── unit/                     # fast, deterministic, no camera/model required*
│   ├── test_camera.py
│   ├── test_config_loader.py
│   ├── test_detector.py
│   ├── test_filter*           (in test_detector / test_phase4_logic)
│   ├── test_tracker.py
│   ├── test_depth.py
│   ├── test_fusion.py
│   ├── test_spatial.py
│   ├── test_priority.py
│   ├── test_temporal.py
│   ├── test_alert_manager.py
│   ├── test_audio.py
│   ├── test_phase4_logic.py   # pure evaluation-logic tests (no camera)
│   ├── test_phase14.py        # performance-monitor + optimization guards
│   ├── test_phase15.py        # false-positive reduction (area gate)
│   └── test_phase16.py        # final-evaluation harness assertions
├── integration/
│   └── test_integration.py    # full pipeline with mocked camera/detector
└── evaluation/                # runnable harnesses (not all are pytest suites)
    ├── phase4_eval.py          # interactive physical detection eval (manual)
    ├── fp_test.py              # empty-scene false-positive measurement
    ├── fn_test.py              # empty-scene sanity + optional static-image
    ├── phase15_fp_replay.py    # audited Phase 15 before/after replay (pytest)
    ├── phase16_pipeline_eval.py# Phase 16 Layer 2 synthetic harness
    ├── depth_test.py           # stub (placeholder)
    └── tracking_test.py        # stub (placeholder)
```

\* Some tests are gated behind markers and are skipped/deselected unless the
required hardware is present (see below).

---

## Markers

Defined in `pyproject.toml`:

| Marker | Meaning |
|---|---|
| `requires_camera` | Needs a physical webcam connected. |
| `requires_model` | Needs the YOLO model loaded. |
| `requires_audio` | Produces actual audio through speakers. |

Deselect hardware tests with `-m "not requires_camera"`.

---

## Running tests

```powershell
# Full non-hardware regression (recommended default):
.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/evaluation/phase15_fp_replay.py -q -m "not requires_camera"

# Everything (will run/skip hardware tests depending on your webcam):
.venv\Scripts\python.exe -m pytest

# A single module:
.venv\Scripts\python.exe -m pytest tests/unit/test_phase16.py -v

# Phase 16 Layer 2 as a script (not pytest):
.venv\Scripts\python.exe tests/evaluation/phase16_pipeline_eval.py
```

---

## Current counts

- **Non-hardware regression: 570 passed, 0 failed** (with 38 `requires_camera`
  hardware tests deselected). [MEASURED — this environment]
- The 38 deselected tests are camera/timing-dependent hardware checks (e.g. raw
  webcam FPS), which vary with the physical device and are not logic tests.

> The exact pass count reflects this repository state and environment. Re-run
> the command above to reproduce it. If a `requires_camera` test runs on your
> machine, its result depends on your webcam/driver and is not a logic failure.

---

## Notes

- `phase4_eval.py` is an **interactive** physical-evaluation runner (needs a
  camera and a person to arrange scenes); its pure logic functions are covered
  by `test_phase4_logic.py`.
- `depth_test.py` and `tracking_test.py` are intentional **stubs** — depth and
  tracking behaviour are covered by `test_depth.py`, `test_fusion.py`,
  `test_tracker.py`, and the Phase 16 synthetic harness.
- Phase 16 Layer 2 correctness is **[SYNTHETIC-DETERMINISTIC]** — it validates
  pipeline logic by construction, not real-world accuracy.
