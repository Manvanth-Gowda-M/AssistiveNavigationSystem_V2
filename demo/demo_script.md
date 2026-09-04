# Live Demonstration Runbook (Operator)

Step-by-step script for the **physical** demonstration. All results below are
**[NOT RUN]** until an operator executes them on real hardware. Do **not**
fabricate outcomes — record exactly what happens, including failures and
misses (they are expected of a research prototype).

## Before you start (safety)

- Use a **safe indoor space**; a **sighted assistant** must be present.
- State aloud to observers: *this is a research prototype, not a certified
  mobility aid, not a replacement for a cane/guide dog/human assistance;
  depth is relative not metric; doors/stairs are not reliably detected.*
- Confirm webcam and speaker/headphones work.
- Use only **reliably detectable** classes for positive demos: **person**
  (most reliable), and where available **chair**, **bottle**, **laptop**.
  Do **not** stage doors/stairs as working detections.

## Setup

```powershell
.venv\Scripts\activate
python -m assistive_navigation            # debug window
#   or: python -m assistive_navigation --headless   # audio-only
```

Record the startup banner (camera / detector / depth / tracker / audio /
**SYSTEM READY**) into `demo/screenshots/` or `demo/startup.txt`.

---

## Scenario table (fill the Result column during the demo)

| ID | Scenario | Physical setup | Expected behavior | Observe | Record | Result |
|----|----------|----------------|-------------------|---------|--------|--------|
| A | Person centered | Stand centered ~1.5–2 m | Confirmed `person`; direction "ahead"; spoken alert | box + "person ahead, …" | screenshot + note | **[NOT RUN]** |
| B | Person left | Stand in left third | direction "on your left" | overlay + speech | screenshot | **[NOT RUN]** |
| C | Person right | Stand in right third | direction "on your right" | overlay + speech | screenshot | **[NOT RUN]** |
| D | Varying proximity | Walk closer then away | proximity band changes; escalation alert when closer | proximity label + speech | screencap/note | **[NOT RUN]** |
| E | Static object confirmation | Place chair/bottle, keep still | no alert until confirmation frames reached, then one alert | delayed first alert | frame-count note | **[NOT RUN]** |
| F | Multiple objects | Person + chair together | one alert/cycle; highest nav_score chosen | both boxes, one spoken | screenshot | **[NOT RUN]** |
| G | Object movement | Person crosses frame | same track_id persists; direction updates | stable id | screencap | **[NOT RUN]** |
| H | Brief occlusion | Step behind object briefly (< max_missed) | track resumes same id | id persists | note | **[NOT RUN]** |
| I | Background/difficult scene | Cluttered static background, no target | ideally no confirmed alert; some large-area FP may remain | few/no alerts | note | **[NOT RUN]** |
| J | Empty scene / FP observation | Clear scene 60 s (see command below) | measured accepted-FP rate | CSV | `data/evaluation/fp_test_<ts>.csv` | **[NOT RUN]** (measurement, not pass/fail) |
| K | Startup/shutdown | Launch then quit (Q / Ctrl+C) | clean banner + clean shutdown | terminal | `demo/startup.txt` | **[NOT RUN]** |

> Scenarios A–I are **[OPERATOR-LIVE]** qualitative observations, **not** formal
> accuracy tests. Scenario J is a **measurement**, not pass/fail. Any scenario
> you do not run stays **[NOT RUN]**.

---

## Audio verification checklist (live) — all **[NOT RUN]**

- [ ] TTS initializes at startup (or documents graceful fallback if unavailable).
- [ ] Alert wording matches templates (e.g. "person ahead, close").
- [ ] No overlapping speech (one alert per cycle; queue serializes).
- [ ] Duplicate suppression: a static unchanged object is not re-announced within cooldown.
- [ ] Escalation: moving closer produces a fresh "now …" alert.
- [ ] Direction wording: ahead / on your left / on your right.
- [ ] Proximity wording: very close / close / nearby.
- [ ] Graceful audio-failure: if speakers/TTS are disabled, the pipeline keeps running silently.

> The deterministic decision logic behind these is already covered by Phase 16
> Layer 2 **[SYNTHETIC-DETERMINISTIC]**. This checklist verifies the **live
> spoken output**, which is **[OPERATOR-LIVE]**.

---

## Live performance runs (optional) — **[NOT RUN]**

```powershell
# 1. Run ~60 s headless, then quit (Ctrl+C):
python -m assistive_navigation --headless

# 2. Immediately capture the perf log (overwritten by the next run):
.venv\Scripts\python.exe tools/phase16_final_eval.py --capture-perf logs/performance.csv
#    -> data/evaluation/phase16_perf_<timestamp>.csv

# Repeat >= 3 times, then analyse all runs together:
.venv\Scripts\python.exe tools/phase16_final_eval.py --perf-csv <runA.csv> --perf-csv <runB.csv> --perf-csv <runC.csv>
```

Record: FPS mean/median/min/max, total/detector/depth latency, depth inference
count + observed interval, **system-wide** CPU, **process** RSS. Compare to the
Phase 13–14 baseline range (11.6–12.5 FPS). A single low run is **not** a
regression.

## Live empty-scene false-positive run (optional) — **[NOT RUN]**

```powershell
python tests/evaluation/fp_test.py --duration 60 --no-display
#    -> data/evaluation/fp_test_<timestamp>.csv   (a MEASUREMENT, not pass/fail)
```

---

## After the demo

- Save screenshots/notes/CSVs into `demo/` (e.g. `demo/screenshots/`,
  `demo/live_perf/`, `demo/live_fp/`).
- Update the Result cells above and change the corresponding labels from
  **[NOT RUN]** to **[OPERATOR-LIVE]** in `demo/README.md` and
  `docs/final_report.md`.
- Do not overwrite the Phase 4/15/16 baseline artifacts.
