# Final Report — Assistive Navigation System V2

Research prototype. Final validation & demonstration summary (Phase 18).

> **Evidence labels** used throughout — do not read any number without its label:
> **[MEASURED]** (real captured data) · **[SYNTHETIC-DETERMINISTIC]** (correct by
> construction; validates logic, not real-world accuracy) · **[OPERATOR-LIVE]**
> (requires a live session) · **[NOT RUN]** (not executed; never fabricated) ·
> **[DESIGN TARGET]** (aspiration/starting estimate) · **[UNVALIDATED ASSUMPTION]**.
>
> This project makes **no** claims of precision, recall, false-negative rate,
> mAP, accuracy, or generalization — no labelled dataset exists to support them.

---

## 1. Problem

People with limited vision benefit from lightweight, low-cost awareness of
nearby obstacles. Commercial aids are costly and often need special hardware.
This project explores how far a **purely software, CPU-only, single-webcam**
prototype can go using free/open-source components — and documents where it
falls short.

## 2. Objectives

- Real-time-ish operation on a CPU-only laptop (no GPU/CUDA). [DESIGN TARGET]
- Free/open-source components only.
- Detect common indoor navigation objects and speak concise alerts.
- Reduce obvious false positives without hiding genuine objects.
- Be honest and auditable: every threshold documented; every number labelled.

## 3. Architecture

Single-process, CPU-only, 11-stage pipeline orchestrated by
`AssistiveNavigationPipeline` (`src/assistive_navigation/main.py`). Each stage
adds fields to a dataclass and passes the rest through unchanged; stages are
isolated by per-stage exception handling. Full detail and diagram:
[`architecture.md`](architecture.md).

```
Camera → Detector → Filter → Tracker → Depth(every N) → Fusion →
Spatial → Priority → Temporal → AlertManager → AudioQueue → TTS
(PerfMonitor wraps every frame; DebugView optional)
```

## 4. Implemented features

- Webcam capture (OpenCV) with FPS measurement.
- Object detection: YOLO11n via **Ultralytics on PyTorch** (`.pt` weights, CPU).
- Confidence + navigation-class filtering with recorded rejection reasons.
- Multi-object tracking: **ByteTrack** (stable `track_id`s, coasting).
- Relative depth: **Depth Anything V2 Small** (float32 ONNX) via ONNX Runtime,
  every `depth_update_interval` (default 5) frames.
- Proximity classification (FAR/MEDIUM/CLOSE/VERY_CLOSE) — **relative**. [UNVALIDATED ASSUMPTION]
- Direction (LEFT/CENTER/RIGHT) from normalized bbox center.
- Navigation priority scoring (`nav_score`) with configurable weights. [DESIGN TARGET]
- Temporal confirmation state machine (5 gates).
- Alert manager (one alert/cycle, dedup, escalation) + non-blocking TTS.
- Performance instrumentation (timing, system-wide CPU, process RSS → CSV).

## 5. Evaluation methodology

Three clearly separated layers (Phase 16), plus the physical detection
baseline (Phase 4) and the false-positive-reduction study (Phase 15). No
labelled dataset → no precision/recall/mAP. Methodology detail:
[`evaluation.md`](evaluation.md).

## 6. Phase 4 — physical detection baseline  [MEASURED]

Interactive physical evaluation on one machine/room/lighting. Key observed
behaviour (from `data/evaluation/phase4_report.txt`, preserved unchanged):

- `person` detected reliably (30/30 frames in the person scenario; mean
  confidence ≈ 0.91). [MEASURED]
- `chair`, `laptop`, `bottle` detected well in their scenarios; `backpack`
  weak (only in multi-object context). [MEASURED]
- Empty-scene false positives: ≈ 203 accepted detections over ~60 s
  (~202.9/min), dominated by background misclassified as `person`. [MEASURED]
- `door`, `doorway`, `stairs`: **not COCO classes → not reliably detected**
  (documented capability gap, safety-critical). [MEASURED]
- `couch`: **NOT TESTED** (no real couch was available). [NOT RUN]

These are scene-specific observations, **not** a generalized accuracy benchmark.

## 7. Phase 15 — false-positive reduction  [MEASURED]

An offline analysis of the Phase 4 data recomputed bounding-box area for every
accepted detection and separated genuine vs false-positive distributions, then
swept the temporal area gate. The single calibrated change was
`temporal.min_area_norm` **0.02 → 0.15**:

- False positives suppressed: **72/303 → 215/303** (23.8% → 71.0%). [MEASURED]
- Genuine objects retained: **114/114 (0 lost)**. [MEASURED]
- Dominant empty-scene background `person` FP: **203 → 22** accepted. [MEASURED]
- No change to `confidence_threshold` (0.35) or detector/tracker/depth/TTS.

Not solved by this gate (documented): class-confusion mislabels; the
door/stairs COCO gap; and the ~29% of empty-scene FP whose area overlaps the
genuine size range. Reproduce: `tools/phase15_fp_analysis.py` +
`pytest tests/evaluation/phase15_fp_replay.py`.

## 8. Phase 16 — three-layer evaluation

- **Layer 1 — detection-level** (offline, reuses Phase 4 CSV read-only):
  reproduces the Phase 15 numbers exactly (see §7). [MEASURED]
- **Layer 2 — synthetic integrated pipeline** (deterministic, by construction):
  73 checks (direction over ≥21 positions + boundaries; proximity bands +
  monotonicity; temporal confirmation; area gate @0.15; tracking persistence &
  coasting; alert correctness/triggers/text/selection/dedup; pipeline-internal
  alert latency **excluding TTS/audio**; multi-object; partial small-area;
  difficult FP profile) — all pass. Validates pipeline **logic**, not
  real-world accuracy. [SYNTHETIC-DETERMINISTIC]
- **Layer 3 — hardware/live**: capture/analysis mechanism implemented
  (FPS mean/median/min/max, latencies, depth interval, system-wide CPU,
  process RSS, live FP). **NOT RUN** — no hardware executed; no results
  fabricated. [NOT RUN]

Methodology, checks, and PASS/FAIL gates: [`evaluation.md`](evaluation.md).

## 9. Phase 17 — reproducibility

Documentation corrected to match the verified environment (README, five filled
doc stubs, usage + reproducibility guides, license inventory incl.
torch/torchvision). Dependency manifest reconciled to the installed versions
(no version changes). Checklists provided:
[`reproducibility.md`](reproducibility.md).

## 10. Phase 18 — final demonstration status

- **Software verification: 570 non-hardware tests pass, 0 failed** (38
  `requires_camera` hardware tests deselected). Protected files verified
  byte-identical to the Phase 17 baseline. [MEASURED — see `../demo/`]
- **Live webcam demonstration** (scenarios A–K): **[NOT RUN]** — operator
  runbook provided in [`../demo/demo_script.md`](../demo/demo_script.md).
- **Live audio verification**: **[NOT RUN]** (deterministic decision logic
  already covered by Phase 16 Layer 2 [SYNTHETIC-DETERMINISTIC]).
- **Live empty-scene FP measurement**: **[NOT RUN]**.

## 11. Performance status

- Depth interval adherence and per-frame instrumentation verified in tests.
  [MEASURED — Phase 14 tests]
- Phase 13/14 measured end-to-end baseline: ≈ **11.6–12.5 FPS** on the
  reference CPU (i5-12450H). [MEASURED — prior phases]
- A **fresh Phase 18 live performance run**: **[NOT RUN]**. When executed via
  `tools/phase16_final_eval.py --capture-perf`, compare against the 11.6–12.5
  FPS baseline; a single low run is not a regression. No live FPS is fabricated.

## 12. Limitations

- **Relative (not metric) depth** — proximity is ordinal. [UNVALIDATED ASSUMPTION]
- **Class-confusion mislabels** (e.g. table → bottle). [MEASURED — Phase 4]
- **Doors/stairs not detectable** (outside COCO classes). [MEASURED — Phase 4]
- **Remaining large-area false positives** overlap the genuine size range. [MEASURED — Phase 15]
- **`couch` NOT TESTED**. [NOT RUN]
- **Single-environment evaluation** — does not generalize.
- Proximity/cooldown/confirmation thresholds are documented starting estimates. [DESIGN TARGET]
- Layer 2 correctness is by construction (synthetic), not real-world accuracy. [SYNTHETIC-DETERMINISTIC]
- Alert latency measured is pipeline-internal and excludes TTS/audio playback.

## 13. Safety disclaimer

This is a **research and educational prototype — NOT a certified assistive
device**. It can and will miss obstacles and produce false detections. It is
**not** a substitute for a mobility aid, white cane, guide dog, or human
assistance, and **not** a replacement for a trained mobility aid/device. Do not
rely on it in traffic, near stairs, on roads, or anywhere a missed detection
could cause injury. Intended for controlled indoor testing with a sighted
person present.

## 14. Licensing

- Project source: **MIT** (`LICENSE`).
- YOLO11n / Ultralytics / ByteTrack: **AGPL-3.0** (satisfied by open-source use).
- Depth Anything V2 **Small**: **Apache-2.0** (Small variant only; Base/Large/
  Giant are CC-BY-NC-4.0 and are **not** used).
- torch, torchvision: **BSD-3-Clause**. ONNX Runtime / pyttsx3 / PyYAML: **MIT**.
  OpenCV: **Apache-2.0**. psutil / numpy: **BSD**.
- Full inventory: [`licenses.md`](licenses.md).

## 15. Future work

- Calibrated/metric depth for true distances.
- A door/stair detector to close the COCO gap.
- A labelled dataset to enable formal precision/recall/mAP evaluation.
- Multi-environment testing for generalization.
- Optional neural TTS (e.g. Piper) for clearer speech.

---

## Appendix — how to reproduce this report's evidence

```powershell
# Software verification (expect 570 passed):
.venv\Scripts\python.exe -m pytest tests/unit tests/integration tests/evaluation/phase15_fp_replay.py -m "not requires_camera"

# Phase 15 reproduction (offline):
.venv\Scripts\python.exe tools/phase15_fp_analysis.py
.venv\Scripts\python.exe -m pytest tests/evaluation/phase15_fp_replay.py

# Phase 16 Layers 1+2 + report (offline):
.venv\Scripts\python.exe tools/phase16_final_eval.py

# Live pieces (operator): see demo/demo_script.md  ([NOT RUN] until executed)
```
