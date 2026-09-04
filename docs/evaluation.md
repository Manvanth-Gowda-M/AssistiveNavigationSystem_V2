# Detection Evaluation — Phase 4
# Assistive Navigation System V2

---

## Purpose

Phase 4 measures how YOLO11n actually behaves on this specific computer,
webcam, and environment. It does not prove or claim accuracy in general.

Results describe observed behavior during the evaluation session. They
cannot be generalised to other rooms, lighting conditions, or cameras.

---

## Safety Notice

This system is a **research prototype**. Detection results from this
evaluation do not imply the system is suitable for independent navigation
or that it can replace a mobility aid, white cane, guide dog, or human
assistance. All findings are prototype measurements only.

---

## Methodology

### Evaluation approach

Standard computer vision evaluation metrics (mAP, Precision, Recall, F1)
require a labelled test image dataset. No such dataset is available for
this project. Therefore, Phase 4 uses **scenario-based evaluation with
live camera and declared ground truth**.

The evaluator (user) places objects in front of the camera one at a time.
The evaluation script runs inference for a defined number of frames and
records every detection. Ground truth is declared at the scenario level
(e.g. "person is in frame"), not per-frame.

### Conservative terminology

Because we lack formal labelled data, we use conservative terminology:

| Term used | What it means |
|---|---|
| correct-class observation | Expected class appeared in accepted output |
| wrong-class observation | A different class appeared in accepted output |
| no-detection observation | Nothing appeared in accepted output |
| FP observation | An accepted detection in a declared-empty scene |
| ignored-class suppression | A known FP class correctly blocked by filter |
| gap observation | Any detection in a COCO-gap scenario (door/stairs) |

We do NOT use "precision", "recall", or "mAP" because the methodology
does not support those metrics.

### Three-layer detection model

Every detection goes through three visible layers before being logged:

```
Raw (>0.15 conf)  ──► Confident (>0.35)  ──► Navigation list  ──► Accepted
                             ↓                       ↓
                       low_confidence          class_ignored
                       (logged)                class_not_in_list
                                               (all logged)
```

All rejected detections are recorded with reason codes. Nothing is
silently discarded.

---

## How to Run the Evaluation

### Prerequisites

```
# Activate the virtual environment (Windows)
.venv\Scripts\activate

# Verify Phase 3 tests still pass
python -m pytest tests/unit/test_detector.py -v --tb=short
```

### Run the full interactive evaluation

```
python tests/evaluation/phase4_eval.py
```

With options:
```
# Run with 60 frames per scenario (slower but more data)
python tests/evaluation/phase4_eval.py --frames 60

# Run with 120-second empty-scene test (Category 3)
python tests/evaluation/phase4_eval.py --fps-duration 120

# Run only specific categories
python tests/evaluation/phase4_eval.py --categories 1,2

# List all scenario IDs
python tests/evaluation/phase4_eval.py --list-scenarios

# No display window (headless)
python tests/evaluation/phase4_eval.py --no-display
```

Output files:
- `data/evaluation/phase4_results.csv` — raw per-frame detection data
- `data/evaluation/phase4_report.txt` — human-readable summary

### Run the false positive rate test

```
python tests/evaluation/fp_test.py
python tests/evaluation/fp_test.py --duration 30
python tests/evaluation/fp_test.py --duration 120 --scene empty_wall
```

### Run the empty-scene sanity test

```
python tests/evaluation/fn_test.py
python tests/evaluation/fn_test.py --frames 60
python tests/evaluation/fn_test.py --camera
```

With a known-object sample image:
```
python tests/evaluation/fn_test.py \
  --sample data/samples/person.jpg \
  --expected-class person
```

### Run automated tests (no camera required for most)

```
python -m pytest tests/evaluation/fp_test.py -v -s
python -m pytest tests/evaluation/fn_test.py -v -s
```

---

## Scenario Definitions

### Category 1 — True Positive Validation

Tests each navigation-critical object individually.

| Scenario ID | Object | Expected class |
|---|---|---|
| person_standing | Person standing in frame | person |
| chair_visible | Chair clearly in view | chair |
| couch_visible | Couch/sofa in view | couch |
| dining_table_visible | Table in view | dining table |
| laptop_visible | Open laptop in view | laptop |
| bottle_visible | Bottle in view | bottle |
| backpack_visible | Backpack in view | backpack |

### Category 2 — False Positive Probe (V1 Problem Objects)

Tests objects that caused false positives in the V1 prototype.

| Scenario ID | Object | V1 concern |
|---|---|---|
| usb_charger | USB charger plug | Previously: cell phone |
| pen_or_pencil | Pen or pencil | Previously: baseball bat |
| phone_only | Mobile phone | Previously: remote |
| empty_desk | Empty desk | Previously: phantom detections |

### Category 3 — Empty Scene FP Rate

Measures accepted detections per minute in a declared-empty scene.
Duration configurable (default: 60 seconds).

### Category 4 — Partial Visibility / Occlusion

Tests detection when objects are partially hidden.

| Scenario ID | Description |
|---|---|
| partial_person | Person ~50% behind door frame |
| partial_chair | Chair ~50% behind wall or object |
| edge_of_frame_bottle | Bottle at edge of camera view |

### Category 5 — Multiple Objects Together

Tests simultaneous detection of multiple objects.

| Scenario ID | Objects |
|---|---|
| person_and_chair | Person + Chair in same frame |
| bottle_laptop_backpack | Bottle + Laptop + Backpack |

### Category 6 — Doors and Stairs Gap Analysis

See dedicated section below.

---

## Metric Definitions

| Metric | Definition |
|---|---|
| correct-class observations | Frames where expected class appeared in accepted output |
| wrong-class observations | Frames where a different class was accepted instead |
| no-detection observations | Frames where nothing was accepted |
| FP observations (accepted/min) | Accepted detections per minute in a declared-empty scene |
| ignored-class suppressions | Times a known FP class was blocked by the filter |
| observed detection rate | correct / total (sample-image test only) |
| confidence statistics | min/max/mean/stdev of accepted detection confidence values |

---

## Limitations of This Evaluation

1. **No labelled dataset.** Formal mAP, precision, recall, and F1 cannot
   be computed. Results describe what was observed, not statistical accuracy.

2. **Single environment.** Results reflect the tested room, lighting, and
   camera position. Performance in other environments may differ significantly.

3. **Confidence ≠ correctness.** A detection with 0.85 confidence may still
   be the wrong class. Do not treat high confidence as proof of correctness.

4. **No tracking.** Each frame is evaluated independently. Temporal
   filtering (Phase 10) will further reduce false positives in the
   full integrated system.

5. **FP/minute is scene-specific.** The empty-scene FP rate describes the
   specific scene tested. Do not generalise to all environments.

6. **Blank frames are NOT FN tests.** See section below.

7. **Doors and stairs are not detectable.** See section below.

---

## Why Blank Frames Do Not Measure False Negatives

A false negative (missed detection) requires:
- A known object that **should** be detected
- The system failing to detect it

A blank (all-zero) or empty-scene frame contains no objects by definition.
Therefore zero detections on a blank frame is **correct behaviour**, not
evidence of missed detections.

`fn_test.py` is named for project compatibility but implements an
**empty-scene sanity test**, not a false-negative measurement.

Formal missed-detection measurement is only possible when:
1. A labelled sample image (known object, known class) is available
2. The script is run with `--sample <image> --expected-class <class>`

Even then, it measures consistency on a single static image, not
generalisable miss rate.

**`fn_test.py` always sets `is_fn_measurement=False` for blank/empty-scene
runs and explains why in `fn_not_possible_reason`.**

---

## YOLO11n Capability Gap: Doors and Stairs

### What this means

YOLO11n is trained on the COCO dataset, which contains 80 object classes.
**Neither "door", "stairs", "doorway", "staircase", nor "step" is a COCO
class.**

This means the model has no dedicated detector for these objects.

### Why this matters for navigation

Doors and stairs are safety-critical navigation obstacles. A navigating
user must know:
- A door is ahead (to avoid walking into it)
- Stairs are ahead (severe fall risk)

The absence of these classes from YOLO11n's training data is a
**documented architectural limitation** of the current system.

### What YOLO11n may do instead

When shown a door or stairs, YOLO11n may:
- Detect nothing (most likely)
- Misclassify parts of the scene as another class (e.g. door frame as
  "refrigerator", railing as unrelated object)
- Produce a very low-confidence incidental detection

**None of these should be interpreted as reliable door or stair detection.**
A misclassification is not a workaround — it is a different kind of error.

### Observed results from Category 6

> **Status: NOT YET MEASURED**
>
> Run `python tests/evaluation/phase4_eval.py --categories 6` to collect
> results. The table below will be populated after evaluation.

| Scenario | Frames | Detections | Classes detected | Assessment |
|---|---|---|---|---|
| door_gap_analysis | 30 | 6 accepted (scattered) | bottle:2, laptop:1, chair:1, backpack:1, person:1 | No dominant wrong class. Low-count scattered FPs. |
| doorway_gap_analysis | 30 | 0 accepted | NONE | Model produces no confident detection for a doorway. |
| stairs_gap_analysis | 30 | 0 accepted | NONE | Model produces no confident detection for stairs. |

**Corrected note:** The first (invalid) run showed `couch` dominating door
and stairs scenarios. This was caused by the evaluator remaining in frame.
The corrected finding is that YOLO11n **silently ignores** doors and stairs
when no person is present — producing zero accepted detections. This is
less dangerous than wrong-class alerts, but the gap (no warning about a
door or staircase) remains a documented safety limitation.

### Future mitigation options (not in scope for Phase 4)

If door and stair detection is required in future phases:
1. **Open-vocabulary detection** (e.g. YOLOE with text prompts) — can
   detect arbitrary classes described in text, without retraining.
2. **Custom dataset + fine-tuning** — collect and label door/stair images,
   fine-tune YOLO11n. Requires significant effort.
3. **Secondary classifier** — train a lightweight binary classifier
   specifically for door/stair recognition.
4. **Depth-based obstacle detection** — Phase 6 depth estimation may
   implicitly detect large flat obstacles regardless of class.

These options will be evaluated in later phases if needed.

---

## V1 Previously Observed Problem Classes

The following detection errors were observed in the V1 prototype.
Phase 4 Category 2 tests whether they still occur under V2.

### Observed V1 problems

| Object shown | V1 detected as | Root cause |
|---|---|---|
| USB charger | cell phone | Visual similarity at low resolution |
| Pen/pencil | baseball bat | Elongated shape confusion |
| Phone | remote | Similar rectangular shape |
| Empty scene | toothbrush, tie | Model hallucination |

### V2 mitigations in place

The following classes are in the **ignored list** in `config.yaml` and will
never reach accepted navigation output, regardless of confidence:

- `remote` (class 65) — phone/remote confusion
- `baseball bat` (class 34) — pen/bat confusion
- `baseball glove` (class 35) — adjacent class
- `toothbrush` (class 79) — phantom detection
- `tie` (class 27) — phantom detection

These suppressions prevent V1 false positives from reaching the alert system.
However, **they do not fix the underlying model confusion** — the raw
detections may still occur. Phase 4 records whether they do.

### Observed results from Category 2

> **Status: NOT YET MEASURED**
>
> Run `python tests/evaluation/phase4_eval.py --categories 2` to collect
> results. The table below will be populated after evaluation.

| Scenario | Object shown | FP in accepted output | Correctly suppressed | Notes |
|---|---|---|---|---|
| usb_charger | USB charger | **57** (person:30, bottle:26, chair:1) | 0 | cell_phone NOT reproduced (corrected run) |
| pen_or_pencil | Pen/pencil | **42** (person:35, bottle:7) | 0 | baseball_bat NOT reproduced; FPs are background |
| phone_only | Mobile phone | 0 | 3 (tie suppressed) | phone largely undetected at 0.35 threshold |
| empty_desk | Empty desk | **1** (person:1, conf 0.561) | 0 | Very low rate; V1 phantoms not reproduced |

---

## Phase 4 Results — Corrected Physical Evaluation

**Status: 19/20 scenarios measured. 1 scenario NOT TESTED (couch_visible — no real couch available).**

Evaluation date: 2026-09-02. All numbers sourced from
`data/evaluation/phase4_results.csv` (corrected run, real physical objects).

---

### Methodology note — corrected run

The first evaluation run (2026-09-02 13:18) was invalidated because the
evaluator remained in the camera frame during object-only scenarios, causing
the `person` class to dominate detections. The bbox analysis confirmed this:
person bboxes in `chair_visible`, `couch_visible`, and others were
pixel-for-pixel identical to `person_standing`. None of those object
detections were valid.

The corrected run (2026-09-02 14:56) was conducted with the evaluator
physically out of frame for all object-only scenarios. The person detections
that remain in some scenarios (e.g. `bottle_visible`, `backpack_visible`) are
clearly distinct from the `person_standing` reference bbox — they are
low-confidence, large-area, top/edge-anchored detections caused by background
patterns, not the evaluator's body.

---

### Overall raw counts — corrected run (19 measured scenarios)

| Observation type | Count |
|---|---|
| correct_class_observation | 242 |
| wrong_class_observation | 210 |
| no_detection_observation | 640 |
| false_positive_observation | 303 |
| ignored_class_suppressed | 3 |
| gap_observation (Cat 6) | 76 |
| empty_scene_correct | varies |
| **NOT TESTED** | couch_visible (1 scenario) |

---

### Category 1 — True Positive Validation Results

| Scenario | Frames | Correct-class | Wrong-class | No-det | Accepted classes | Conf mean |
|---|---|---|---|---|---|---|
| person_standing | 30 | **30** | 0 | 0 | person:30 | 0.913 |
| chair_visible | 30 | **26** | 2 | 24 | chair:26, person:2 | 0.738 |
| **couch_visible** | — | — | — | — | **NOT TESTED** | — |
| dining_table_visible | 30 | 0 | **30** | 57 | bottle:29, chair:1 | 0.848 |
| laptop_visible | 30 | **29** | 29 | 32 | laptop:29, person:28, bottle:1 | 0.749 |
| bottle_visible | 30 | **29** | 33 | 64 | person:32, bottle:29, laptop:1 | 0.663 |
| backpack_visible | 30 | 0 | **42** | 99 | person:32, chair:9, bottle:1 | 0.416 |

**Key findings — Category 1 (corrected):**

- `person`: Detected correctly in 30/30 frames at high confidence (mean 0.913,
  stdev 0.005). The most reliable navigation class in this system.

- `chair`: Detected correctly in 26/30 frames (87% frame coverage). Confidence
  was variable (mean 0.738, stdev 0.170, min 0.350). The 2 wrong-class
  observations were `person` at low confidence — plausibly the model interpreting
  the chair legs/back as a partial person. The 4 no-detection frames suggest
  intermittent detection at the confidence floor.

- `couch / sofa`: NOT TESTED. No real couch available. Will remain not tested
  until a physical couch is available for evaluation.

- `dining table`: Never detected as `dining table` in 30 frames. The dominant
  detected class was `bottle` (29 accepted, mean conf 0.848, bbox at ~437px x1).
  Inspecting the bboxes, these `bottle` detections are positioned consistently
  in a small region (~437–500px x1, ~300–465px y1) — they represent objects ON
  the table surface (cups, bottles, utensils), not the table itself.
  `dining table` is not in the COCO training distribution for this viewing angle.
  The 57 no-detection rows confirm the table itself was not detected.

- `laptop`: Detected correctly in 29/30 frames (97% frame coverage). The 28
  `person` wrong-class observations in the same scenario (avg_x1=426, avg_y1=287,
  confined to lower-right quadrant, avg_conf=0.565) are background pattern FPs
  at the laptop's surroundings, not the evaluator. Detection quality is good.

- `bottle`: Detected correctly in 29/30 frames (97% frame coverage). The 32
  `person` wrong-class observations (avg_x1=14, avg_y1=12, conf ~0.37–0.53) are
  large low-confidence bboxes spanning most of the frame — these are background
  texture FPs at the confidence floor, not a person. Detection quality is good.

- `backpack`: Never detected as `backpack` in the standalone scenario. The 42
  wrong-class observations were `person` (32) and `chair` (9) — both background
  elements visible near the backpack. Backpack detection is unreliable alone.

---

### Category 2 — False Positive Probe Results (corrected)

| Scenario | Frames | FP obs (accepted) | FP/min | Accepted classes |
|---|---|---|---|---|
| usb_charger | 30 | 57 | 2,795 | person:30, bottle:26, chair:1 |
| pen_or_pencil | 30 | 42 | 1,913 | person:35, bottle:7 |
| phone_only | 30 | 0 | 0.00 | person:29, bottle:9 |
| empty_desk | 30 | 1 | 45.8 | person:1 |

**Key findings — Category 2 (corrected):**

- `USB charger`: The V1 concern (charger → cell phone) was **not reproduced** in
  this corrected run. No `cell phone` detections appeared. Instead: `person` (30,
  low-conf, large top-edge bboxes — background FP) and `bottle` (26, mean conf
  ~0.60 — the charger cable/plug shape is triggering bottle detections). The
  charger itself is not detected as any correct class. The high FP/min rate is
  driven by background pattern detections, not the charger specifically.

- `Pen/pencil`: `baseball bat` V1 concern was **not reproduced**. Instead:
  `person` (35, low-conf background FPs) and `bottle` (7, low conf). The pen
  itself is not detected as any specific class. No dangerous misclassification
  to a navigation category was observed.

- `Mobile phone`: `remote` V1 concern was **not reproduced** — `tie` was
  correctly suppressed (3 ignored events). `cell phone` was never accepted.
  The phone is largely undetected (64 no-detection rows). `person` (29) and
  `bottle` (9) accepted detections are low-conf background FPs.
  **Note:** The phone scenario had `correct_class_observations: 0` because
  `cell phone` was never accepted — inspection shows `cell phone` appears in
  raw detections but falls below the confidence threshold, or the phone angle/
  placement made it unrecognisable at 0.35 threshold.

- `Empty desk`: Only 1 accepted FP (`person`, conf 0.561) in 30 frames.
  FP rate 45.8/min is the base room background rate, not a specific object
  misclassification. V1 phantom classes (toothbrush, tie) were not reproduced.

---

### Category 3 — Empty Scene FP Rate (corrected, 60 seconds)

| Scene | Duration (s) | Frames | Accepted total | Accepted/min |
|---|---|---|---|---|
| empty_scene_fp_rate | 60.03 | 922 | 203 | **202.9** |

Accepted class breakdown:

| Class | Count | Notes |
|---|---|---|
| person | 200 | Avg conf 0.792, bbox confined to lower-right quadrant (x1~409, y1~250–480, width~190px). Consistent small region. Likely a background element (picture frame, poster, or similar pattern) being interpreted as a partial person. |
| chair | 2 | Low-count background element |
| laptop | 1 | Low-count background element |

**Key finding — Category 3 (corrected):**
FP rate dropped from ~898/min (first run, evaluator partially in frame) to
**202.9/min** in the corrected run. The 200 `person` FPs come from a single
consistent region in the lower-right quadrant (confirmed by bbox clustering:
avg x1=409, y1=250, x2=599, y2=480, conf mean 0.792). This is a specific
background element in the test room being persistently misclassified as a
partial person at moderate-to-high confidence.

This finding is critical for Phase 10 (temporal confirmation): even in a
"genuinely empty" scene, the detector produces ~200 accepted
`person` detections per minute from a static background feature.
**Temporal confirmation with sufficiently high `CONFIRMATION_FRAMES` is
the primary mitigation. A background feature is static — it will pass
temporal confirmation. A minimum-bounding-box-size filter is also needed.**

---

### Category 4 — Partial Visibility / Occlusion Results (corrected)

| Scenario | Frames | Correct-class | Wrong-class | No-det | Notes |
|---|---|---|---|---|---|
| partial_person | 30 | 0 | 0 | **30** | No detections at all |
| partial_chair | 30 | 0 | 0 | **30** | No detections at all |
| edge_of_frame_bottle | 30 | 0 | 0 | **30** | No detections at all |

**Key finding — Category 4 (corrected):**
All three occlusion scenarios produced zero accepted detections across all
30 frames. The one sub-threshold raw detection seen was `tv` at conf 0.154
in `partial_person`. This means:

- A person occluded ~50% falls below the 0.35 threshold entirely.
- A chair occluded ~50% produces no accepted detection.
- A bottle at the frame edge produces no accepted detection.

**This is a significant finding.** Partial visibility completely suppresses
detection for these objects at the current confidence threshold. For navigation
use, this means the system will not alert about partially hidden obstacles.
Phase 10 temporal confirmation cannot compensate for zero raw detections.

---

### Category 5 — Multiple Objects Results (corrected)

| Scenario | Expected | Frames | Correct-class | Wrong-class | No-det | Accepted classes |
|---|---|---|---|---|---|---|
| person_and_chair | person+chair | 30 | 30 | 1 | 33 | person:29, chair:1, dog:1 |
| bottle_laptop_backpack | bottle+laptop+backpack | 30 | 98 | 35 | 177 | bottle:42, laptop:29, backpack:27, person:21, chair:13, book:1 |

**Key finding — Category 5 (corrected):**

- `person_and_chair`: Person was detected in 29/30 frames (consistent with
  standalone result). Chair was detected in only 1 frame — adding a person
  to the scene suppressed chair detection. `dog` appeared once (conf 0.361)
  — a marginal low-confidence misclassification.

- `bottle+laptop+backpack`: This is the **first scenario where backpack was
  detected** — 27 accepted detections, mean conf ~0.79, bbox lower-right
  quadrant. This confirms backpack detection requires a multi-object context
  or specific positioning. Bottle (42) and laptop (29) also detected. The 35
  wrong-class observations were `person` (21) and `chair` (13) — background
  FPs. No `backpack` was detected in the standalone scenario.

---

### Category 6 — COCO Gap Analysis Results (corrected)

| Scenario | Frames | Gap obs. | Accepted classes |
|---|---|---|---|
| door_gap_analysis | 30 | 14 | bottle:2, laptop:1, chair:1, backpack:1, person:1 |
| doorway_gap_analysis | 30 | 59 | **NONE** |
| stairs_gap_analysis | 30 | 3 | **NONE** |

**Key finding — Category 6 (corrected):**
The corrected run shows a substantially different result from the first run.
The first run showed `couch` dominating door and stairs — this was caused by
the evaluator being in frame. With the evaluator absent:

- `Door`: 14 gap observations, scattered low-count misclassifications
  (bottle:2, laptop:1, chair:1, backpack:1, person:1). No dominant wrong class.
- `Doorway`: 59 gap observations — 0 accepted detections. The model produces
  no confident detection for a doorway.
- `Stairs`: 3 gap observations — 0 accepted detections.

**Corrected conclusion:** YOLO11n produces **no reliable detection at all**
for doors or stairs when the evaluator is not in frame. The `couch` pattern
seen in the first run was a contamination artefact. The true finding is that
the model simply ignores these structures — which is consistent with them not
being COCO classes. Zero detections, not wrong detections.

---

### Ignored-class suppression summary (corrected run)

| Suppressed class | Count |
|---|---|
| tie | 3 |

Only 3 suppression events observed in the corrected run (all `tie` — from the
phone scenario, where the phone edge/shape triggered a low-confidence `tie`
detection). The V1 phantom classes (toothbrush×38, tie×12 in the first run)
were substantially reduced in the corrected run, confirming they were largely
caused by the evaluator being in frame.

---

### Overall findings and implications — corrected

**What works reliably:**

| Object | Detection rate | Confidence | Notes |
|---|---|---|---|
| `person` (full view) | 30/30 = 100% | mean 0.913 | Highly reliable |
| `laptop` (full view) | 29/30 = 97% | varies | Good; co-detects background as person |
| `bottle` (full view) | 29/30 = 97% | varies | Good; co-detects background as person |
| `chair` (full view) | 26/30 = 87% | mean 0.738 | Acceptable; confidence variable |
| `backpack` (multi-object) | 27/30 frames | mean ~0.79 | Only in multi-object context |

**What does not work or is unreliable:**

| Finding | Evidence |
|---|---|
| `dining table` never detected | 0/30 correct; objects ON table detected as bottle instead |
| `backpack` alone never detected | 0/30 correct in standalone; only detected in multi-object group |
| Partial visibility suppresses all detection | 0/90 correct across all 3 occlusion scenarios |
| Empty scene has 203 FP/min from static background | Persistent lower-right region misclassified as `person` at conf ~0.79 |
| `phone` largely undetected alone | 0/30 correct-class observations |

**What changed from the first (invalid) run:**

| Claim in first run | Corrected finding |
|---|---|
| stairs → couch at 0.689 conf (safety critical) | Corrected: stairs → 0 accepted detections. Couch was from evaluator in frame. |
| door → couch/person/chair | Corrected: door → few scattered low-conf FPs. No dominant wrong class. |
| empty scene FP rate ~898/min | Corrected: ~203/min — still high, but specific to one background region |
| chair, couch, table, backpack never detected | Corrected: chair detected 87%, laptop 97%, bottle 97%, backpack in multi-object |

**Direct implications for subsequent phases:**

- **Phase 10 (Temporal confirmation):** `CONFIRMATION_FRAMES` must be tuned
  against the ~203 FP/min background rate. At ~15 FPS detection, 203/min = ~3.4
  FPs per second. A 10-frame confirmation window at 15 FPS = 0.67s — the
  probability of the same background region producing 10 consecutive frames is
  high. A minimum-area filter OR a position-stability filter is needed alongside
  temporal confirmation.
- **Phase 15 (FP reduction):** The persistent background `person` FP is the
  primary target. Minimum bounding-box area relative to frame and position-
  stability checks should be investigated.
- **Doors/stairs:** The corrected finding (0 detections) is actually less
  dangerous than the first finding (wrong-class confident detections). A zero
  detection means silence — a wrong-class detection means a wrong alert.
  Silence is better, but the gap remains a documented limitation.

---

## Files Generated by Evaluation

| File | Contents |
|---|---|
| `data/evaluation/phase4_results.csv` | Per-frame raw detection data |
| `data/evaluation/phase4_report.txt` | Human-readable summary report |
| `data/evaluation/fp_test_<timestamp>.csv` | FP rate measurement output |

CSV columns in `phase4_results.csv`:
```
scenario, category, timestamp, ground_truth,
detected_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
accepted, eval_result, rejection_reason, raw_latency_ms
```

---

*Last updated: Phase 4 implementation. Results section pending manual evaluation.*


---

# Phase 16 — Final System Evaluation

> The Phase 4 section above is preserved verbatim as the detection-level
> baseline. Phase 16 does NOT modify Phase 4 or Phase 15 artifacts. Phase 16
> added evaluation code only; it changed no thresholds and no pipeline module.

## Purpose

Provide a rigorous, honest, end-to-end evaluation of the complete assistive
navigation system, separating what can be measured deterministically from
what depends on live hardware.

## Honesty boundaries (read first)

- **No labelled dataset** exists, so this evaluation makes **no** claims of
  precision, recall, false-negative rate, mAP, accuracy, or generalization.
- **Layer 2 uses synthetic ground truth (by construction):** it validates
  pipeline **logic**, not real-world detection quality.
- Depth is **relative, not metric** — proximity is ordinal.
- **Single environment / camera / lighting.**
- **Alert latency is pipeline-internal and EXCLUDES TTS/audio playback time.**

## Three evaluation layers

### Layer 1 — Detection-level (offline, read-only)
Reuses the existing `data/evaluation/phase4_results.csv` (never modified).
Reproduces the Phase 15 area-gate replay numbers and reports:
- Phase 15 before/after: `min_area_norm` 0.02 → 0.15 suppresses
  72/303 → **215/303** empty-scene-family false positives (71.0%) with
  **114/114 genuine detections retained (0 lost)**.
- Genuine-object retention by class @ 0.15: person 30/30, chair 26/26,
  laptop 29/29, bottle 29/29.
- Accepted false positives by scene: empty_scene 203, usb_charger 57,
  pen_or_pencil 42, empty_desk 1.
- Wrong-class / class-confusion observations (FP-3): 210 — **not** solved by
  the area gate; documented limitation.
- Observed no-detection counts for known-object scenarios — reported as
  **scene-specific observed misses**, explicitly **not** a generalized FN rate.

Tool: `tools/phase16_final_eval.py` (function `layer1_detection`).

### Layer 2 — Synthetic integrated pipeline (deterministic)
`tests/evaluation/phase16_pipeline_eval.py` drives the **real** stage classes
(SpatialReasoner, DepthFusion, NavigationPriorityEngine,
TemporalConfirmationFilter, ObjectTracker, AlertManager) with **constructed**
inputs, so the correct output is known exactly. Each check is tagged
`exact` (deterministic correctness), `boundary` (deterministic edge), or
`heuristic` (threshold-dependent, reported for self-consistency).

Evaluated:
- **Direction** over ≥21 known positions + explicit boundaries + clamp + NaN.
- **Proximity** bands over multiple depth values + **monotonicity** + the full
  `fuse()` path with uniform depth maps.
- **Temporal confirmation** (phantom never confirms; confirms at exactly
  `confirmation_frames`; brief gap resumes; excessive gap expires).
- **Area gate @ 0.15** (genuine min 0.1612 confirms; FP median 0.0645 hard-fails).
- **Tracking** persistence, single-ID under smooth motion, short-occlusion
  coasting, empty-input safety.
- **Alerts**: trigger reasons, alert text formatting, highest-`nav_score`
  selection, **duplicate suppression within cooldown**, alert count vs
  expected, and **pipeline-internal alert latency** (frames to `alert_issued`,
  **excluding** TTS/audio).
- **Multiple objects**, **partial/small-area** around the 0.15 gate, and the
  **difficult/background FP profile** (must never confirm and never alert).

### Layer 3 — Hardware / live camera (operator-run; never fabricated)
`tools/phase16_final_eval.py` captures and analyses real runs:
- `--capture-perf logs/performance.csv` copies each completed run to a
  timestamped `data/evaluation/phase16_perf_<ts>.csv` (the live log is
  overwritten each run, so this preserves it).
- `--perf-csv` analyses one or more captured runs: FPS mean/median/min/max,
  detector/depth/total latency, depth inference count + observed interval,
  **system-wide** CPU (`cpu_percent_system`), and **process** RSS.
- FPS is compared against the Phase 13–14 baseline range (11.6–12.5 FPS);
  a single low run is **never** treated as a phase failure.
- `--live-fp-csv` records a live empty-scene FP run (from `fp_test.py`) as a
  **measurement** (not pass/fail).
- Any layer not executed is explicitly marked **NOT RUN**.

## How to run

```
# Offline layers 1 + 2 + report (no hardware):
.venv/Scripts/python.exe tools/phase16_final_eval.py

# Layer 2 alone:
.venv/Scripts/python.exe tests/evaluation/phase16_pipeline_eval.py

# Deterministic tests:
.venv/Scripts/python.exe -m pytest tests/unit/test_phase16.py -v

# Hardware (operator): run the pipeline ~60s, then capture, repeat >=3x:
python -m assistive_navigation --headless
.venv/Scripts/python.exe tools/phase16_final_eval.py --capture-perf logs/performance.csv
# Live empty-scene FP:
python tests/evaluation/fp_test.py --duration 60 --no-display
```

## PASS/FAIL gates
- All existing logic/integration tests remain passing (570 total with Phase 16).
- Phase 15 replay numbers reproduce exactly.
- Direction: 100% exact match away from boundaries.
- Proximity: 100% band match away from thresholds; monotonicity holds.
- Temporal/tracking/alert deterministic assertions pass.
- Static unchanged object produces zero repeat alerts within cooldown.
- Performance compared to baseline range; single low FPS run never fails the phase.
- Live FP is a measurement, not pass/fail. Anything not executed is NOT RUN.

## Result files (all timestamped; baselines never overwritten)
- `data/evaluation/phase16_report_<ts>.txt`
- `data/evaluation/phase16_pipeline_eval_<ts>.csv`
- `data/evaluation/phase16_perf_<ts>.csv` (when a live run is captured)
- `data/evaluation/phase16_live_fp_<ts>.csv` (when a live FP run is captured)

## Limitations carried forward
Class-confusion mislabels (FP-3); door/stairs COCO gap (FP-4); remaining
large-area false positives that overlap the genuine cluster; `couch_visible`
NOT TESTED; single-environment evaluation; relative (not metric) depth;
Layer 2 correctness is by construction (synthetic); alert latency excludes
TTS/audio playback.

*Phase 16 added evaluation code and documentation only. No thresholds,
detector/depth/tracker/TTS architecture, config, or Phase 4/15 baselines
were changed.*
