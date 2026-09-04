# Usage — Running, Testing, and Evaluation

Covers README items 23–27: running the application, debug/headless modes,
running tests, running evaluation tools, and running the hardware evaluation.

Prerequisite: complete [`installation.md`](installation.md) first (venv,
dependencies, and models).

---

## 23. Running the application

```powershell
.venv\Scripts\activate
python -m assistive_navigation
```

On start the system loads the detector (critical), depth model (optional),
opens the camera (critical), and initializes audio (optional). It prints a
status banner and then processes frames until you quit.

Optional flags:

```powershell
python -m assistive_navigation --headless        # no OpenCV window (audio-only)
python -m assistive_navigation --config PATH      # use a custom config file
```

Exit: press **Q** in the debug window, or **Ctrl+C** in the terminal. The
pipeline shuts down cleanly (stops audio, releases the camera).

## 24. Debug vs headless mode

- **Debug (default):** an OpenCV overlay window shows boxes, class/confidence,
  track IDs, direction, proximity, and FPS. Controlled by the `debug` and
  `pipeline.show_debug_window` sections in `config/config.yaml`.
- **Headless (`--headless`):** no window is opened (useful for servers, CI, or
  audio-only operation). Debug rendering — including the frame copy — is skipped
  for efficiency.

---

## 25. Running tests

```powershell
# Full non-hardware regression (expected: 570 passed, 0 failed):
.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/evaluation/phase15_fp_replay.py -q -m "not requires_camera"

# A single module, verbose:
.venv\Scripts\python.exe -m pytest tests/unit/test_phase16.py -v
```

See [`testing.md`](testing.md) for the full layout, markers, and counts.

---

## 26. Running evaluation tools

**Layer 2 — synthetic pipeline evaluation** (deterministic, no hardware):

```powershell
.venv\Scripts\python.exe tests/evaluation/phase16_pipeline_eval.py
# writes data/evaluation/phase16_pipeline_eval_<timestamp>.csv
```

**Full Phase 16 orchestrator** (Layer 1 offline + Layer 2 + report):

```powershell
.venv\Scripts\python.exe tools/phase16_final_eval.py
# writes data/evaluation/phase16_report_<timestamp>.txt
#   Layer 1 reproduces the Phase 15 replay numbers from phase4_results.csv (read-only)
#   Layer 3 is marked NOT RUN unless you provide hardware data (below)
```

**Phase 15 false-positive analysis / replay** (offline, read-only on Phase 4 CSV):

```powershell
.venv\Scripts\python.exe tools/phase15_fp_analysis.py           # area analysis
.venv\Scripts\python.exe -m pytest tests/evaluation/phase15_fp_replay.py -q
```

All evaluation tools write **new, timestamped** files under
`data/evaluation/`. They never overwrite the Phase 4 baselines
(`phase4_results.csv`, `phase4_report.txt`).

---

## 27. Running the hardware / live evaluation (operator)

These require a real webcam and produce **[OPERATOR-LIVE]** results. If you do
not run them, they remain **NOT RUN** — never fabricated.

**Performance runs (≥3 recommended, ~60s each):**

```powershell
# 1. Run the pipeline headless for ~60 seconds, then quit (Ctrl+C):
python -m assistive_navigation --headless

# 2. Immediately capture the perf log (it is overwritten on the next run):
.venv\Scripts\python.exe tools/phase16_final_eval.py --capture-perf logs/performance.csv
#    -> copies to data/evaluation/phase16_perf_<timestamp>.csv and analyses it

# Repeat steps 1-2 at least three times, then analyse all captured runs:
.venv\Scripts\python.exe tools/phase16_final_eval.py --perf-csv data/evaluation/phase16_perf_A.csv --perf-csv data/evaluation/phase16_perf_B.csv --perf-csv data/evaluation/phase16_perf_C.csv
```

Reported per run: FPS mean/median/min/max, detector/depth/total latency, depth
inference count + observed interval, **system-wide** CPU, and **process** RSS.
FPS is compared to the Phase 13–14 baseline range; a single low run is never
treated as a failure.

**Live empty-scene false-positive run (~60s):**

```powershell
python tests/evaluation/fp_test.py --duration 60 --no-display
# writes data/evaluation/fp_test_<timestamp>.csv (a MEASUREMENT, not pass/fail)
```

**Optional person/chair spot checks:** run the app with a person or chair in
view and confirm a confirmed alert with the correct direction/proximity and no
alert storm. Record qualitatively.
