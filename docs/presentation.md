# Presentation — Assistive Navigation System V2

A ~14-slide speaker-notes outline for a college/research project review and
technical demonstration. Convert to slides as needed. **Only supported results
appear here.** Evidence labels: [MEASURED] · [SYNTHETIC-DETERMINISTIC] ·
[OPERATOR-LIVE] · [NOT RUN] · [DESIGN TARGET] · [UNVALIDATED ASSUMPTION].
No precision/recall/mAP/accuracy/generalization is claimed.

---

## Slide 1 — Title
**Assistive Navigation System V2** — a software-only, CPU-based, single-webcam
navigation prototype. Research prototype (not a certified mobility aid).
*Speaker note:* state the safety boundary up front.

## Slide 2 — Problem
Low-vision users need low-cost awareness of nearby obstacles; commercial aids
are expensive and hardware-heavy. Can a purely software, CPU-only, webcam
prototype help — and where does it fail?

## Slide 3 — Motivation
Free/open-source components; runs on an ordinary laptop; fully offline after
setup; privacy-preserving (local processing, no upload).

## Slide 4 — Objectives
Real-time-ish CPU-only operation [DESIGN TARGET]; detect common indoor objects;
speak concise alerts; reduce false positives without hiding real objects; stay
honest and auditable (every threshold documented, every number labelled).

## Slide 5 — Architecture
Single-process, 11-stage pipeline; each stage enriches a dataclass; per-stage
exception isolation; one config file as the single source of truth.
*Show the block diagram from* `architecture.md`.

## Slide 6 — Core 11-stage pipeline
Camera → Detector → Filter → Tracker → Depth (every N frames) → Fusion →
Spatial (L/C/R) → Priority (nav_score) → Temporal confirmation → Alert manager
→ Audio queue → TTS. PerfMonitor wraps every frame.

## Slide 7 — Key technical components
- YOLO11n via **Ultralytics/PyTorch** (`.pt`, CPU).
- **ByteTrack** tracking (stable IDs, coasting).
- **Depth Anything V2 Small** (float32 ONNX, ONNX Runtime); **relative** depth. [UNVALIDATED ASSUMPTION]
- Direction zones (0.35 / 0.65); proximity thresholds (0.75/0.55/0.35). [DESIGN TARGET]
- Temporal confirmation (5 gates); alert dedup + escalation; non-blocking TTS.

## Slide 8 — False-positive problem + Phase 15 solution
- Problem: empty-scene background misclassified as `person` (~202.9/min). [MEASURED — Phase 4]
- Solution: calibrate one gate, `min_area_norm` 0.02 → 0.15 (evidence-based, not blind).
- Result: FP suppressed **72/303 → 215/303 (71.0%)**, genuine retained **114/114 (0 lost)**;
  dominant empty-scene FP **203 → 22**. [MEASURED — Phase 15]
- Honest: does not fix mislabels or the door/stairs gap.

## Slide 9 — Three-layer evaluation methodology
- Layer 1 detection-level (offline, Phase 4 data). [MEASURED]
- Layer 2 synthetic pipeline (deterministic, by construction). [SYNTHETIC-DETERMINISTIC]
- Layer 3 hardware/live (mechanism implemented). [NOT RUN]
No labelled dataset → no precision/recall/mAP.

## Slide 10 — Results (with evidence labels)
- Software: **570 non-hardware tests pass**. [MEASURED]
- Phase 15 FP reduction as on Slide 8. [MEASURED]
- Layer 2: **73 deterministic checks pass** (direction, proximity + monotonicity,
  temporal, area gate, tracking, alerts, latency-excluding-TTS, multi/partial/
  difficult). [SYNTHETIC-DETERMINISTIC]
- Performance baseline ≈ **11.6–12.5 FPS** (i5-12450H). [MEASURED — Phase 13/14]
- Fresh live FPS / live demo / live audio / live FP: **[NOT RUN]**.

## Slide 11 — Live demonstration (scenarios / status)
Scenarios A–K (person centered/left/right, varying proximity, static-object
confirmation, multiple objects, movement, brief occlusion, difficult scene,
empty-scene FP, startup/shutdown). All **[OPERATOR-LIVE]**, currently **[NOT
RUN]**; runbook in `../demo/demo_script.md`. Reliable demo classes: person
(best), chair/bottle/laptop. **Doors/stairs shown only as a limitation.**

## Slide 12 — Limitations
Relative (not metric) depth; class-confusion mislabels; doors/stairs COCO gap;
remaining large-area FP; `couch` NOT TESTED; single-environment; thresholds are
starting estimates; Layer 2 is by construction; alert latency excludes TTS.

## Slide 13 — Future work
Calibrated/metric depth; door/stair detector; labelled dataset for formal
metrics; multi-environment testing; optional neural TTS.

## Slide 14 — Conclusion + Q&A
A complete, auditable, honest CPU-only prototype: measured detection baseline,
evidence-based FP reduction, deterministic pipeline validation, full docs and
reproducibility. Explicitly bounded: research prototype, not a mobility aid.
*Have `final_report.md`, `evaluation.md`, and the `demo/` evidence ready for
reviewer questions.*
