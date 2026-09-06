# Live Demonstration Session — 2026-09-04 (~12:41–12:43)

**Evidence label: [OPERATOR-LIVE].** This is a single, uncontrolled live run on
the operator's machine. It is **NOT a controlled benchmark** and must not be
read as characteristic performance or as a formal accuracy test. Numbers below
are the actual values printed during the session; nothing is inferred or
invented.

Launch command used (see packaging note at the end):
```
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'src'); from assistive_navigation.main import main; main()"
```

---

## Startup & component initialization  [OPERATOR-LIVE]

Observed startup banner — all components reported OK:

```
  [OK ] Audio       : SAPI5 / Zira
  [OK ] Detector    : yolo11n (CPU)
  [OK ] Depth       : DA-V2-Small (252px, every 5 frames)
  [OK ] Camera      : 640x480 (device 0)
  [OK ] Tracker     : ByteTrack (CPU)
  SYSTEM READY
```

## Alerts observed  [OPERATOR-LIVE]

All four alert trigger reasons were observed live:

- `new_object` — e.g. "person detected ahead", "person on your left, nearby"
- `proximity_escalation` — e.g. "person ahead, now nearby"
- `direction_change` — e.g. "person detected on your right" then "ahead"
- `cooldown_expired` — e.g. "person detected ahead" after the cooldown window

Direction wording observed: **ahead**, **on your left**, **on your right**.
Proximity wording observed: **nearby**, **now nearby**.
Alerts were spaced out (not emitted every frame), consistent with cooldown /
duplicate suppression.

## Scenarios exercised (qualitative, [OPERATOR-LIVE] — NOT accuracy tests)

| ID | Scenario | Observed | Result |
|----|----------|----------|--------|
| A | Person centered | "person detected ahead" / "person ahead, nearby" | Observed working |
| B | Person left | "person detected on your left" | Observed working |
| C | Person right | "person detected on your right" | Observed working |
| D | Varying proximity | multiple `proximity_escalation` → "now nearby" | Observed working |
| G | Movement / direction change | `direction_change` alerts as position changed | Observed working |
| — | Alert trigger coverage | all 4 reasons observed | Observed working |
| — | Duplicate suppression / cooldown | alerts spaced, not per-frame | Observed working |
| K | Startup / shutdown | Ctrl+C → "Pipeline stopped cleanly. System stopped cleanly." | Observed working |

Scenarios **E (static-object confirmation), F (multiple distinct objects),
H (brief occlusion), I (difficult/background)** were not separately isolated in
this session and remain **[NOT RUN]** as discrete checks.

## Performance observations  [OPERATOR-LIVE — single, CPU-contended session]

Printed per-30-frame stats during the session (verbatim ranges):

- FPS: started ~**7.9–8.4**, then declined to ~**5.0–5.7**, with a low of
  **3.4** near the end.
- Detector latency: ~**85–96 ms** early, rising to ~**120–193 ms**.
- Depth latency (when it ran): ~**126–136 ms** early, rising to ~**221–406 ms**.
- CPU: ~**80–100% system-wide** throughout (labelled system-wide, not per-process).
- RAM (process RSS): ~**555–573 MB**.

**Honest interpretation:** this run was **below** the Phase 13/14 measured
baseline of 11.6–12.5 FPS, and latency/ FPS **degraded over time** while CPU was
saturated (80–100% system-wide) — consistent with CPU contention and/or thermal
throttling, not established as a code regression. Per the project rule, **a
single low/ degrading run is NOT treated as a regression.** Characteristic
performance requires the clean repeat captures below, which are **[NOT RUN]**.

## Tracking observation (honest caveat)  [OPERATOR-LIVE]

Track IDs climbed during the session (1 → 5 → 13 → 16 → 21 → 103 → 112),
i.e. the operator was repeatedly lost and re-acquired as a **new** track rather
than held as one stable ID. At the low frame rate of this session, the
between-frame gaps are large enough to break ByteTrack association. Therefore:

- Single-frame **direction / proximity / alert** behavior was demonstrated. ✅
- **Long-term stable-ID persistence and clean brief-occlusion coasting were
  NOT cleanly demonstrated** in this session (IDs fragmented). This is a live
  observation tied to the low FPS, recorded honestly rather than hidden.

## Clean shutdown  [OPERATOR-LIVE]

```
KeyboardInterrupt — stopping pipeline.
Pipeline shutting down ...
Pipeline stopped cleanly.
System stopped cleanly.
```

---

## Still [NOT RUN] (require separate physical execution — not fabricated)

- **Scenario J** — 60-second empty-scene false-positive measurement
  (`python tests/evaluation/fp_test.py --duration 60 --no-display`). **[NOT RUN]**
- **Clean performance captures** (≥3 runs with other apps closed, via
  `tools/phase16_final_eval.py --capture-perf logs/performance.csv`). **[NOT RUN]**
- Discrete scenarios E, F, H, I as isolated checks. **[NOT RUN]**

## Packaging note

This session was launched with an explicit `sys.path` workaround because
`python -m assistive_navigation` did not work at the time of this session.
The packaging/startup fix is a separate unfinished follow-up and is intentionally
not included in this evidence commit.
