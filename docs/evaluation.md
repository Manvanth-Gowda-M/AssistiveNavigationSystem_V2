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
| door_gap_analysis | 30 | 35 accepted | couch:13, person:11, chair:9, cell_phone:2 | **No door detection. Misclassified as furniture/person.** |
| doorway_gap_analysis | 30 | 9 accepted | couch:5, chair:3, person:1 | No doorway detection. Low-frequency misclassifications. |
| stairs_gap_analysis | 30 | 34 accepted | couch:29, chair:4, person:1 | **Stairs→couch at 0.689 mean conf. Consistent and safety-critical.** |

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
| usb_charger | USB charger | **65** (person:42, cell_phone:23) | 0 | V1 concern confirmed: cell_phone appeared |
| pen_or_pencil | Pen/pencil | **74** (person:32, laptop:20, book:14, cell_phone:8) | 0 | baseball_bat NOT observed; new classes instead |
| phone_only | Mobile phone | 0 | 5 (remote suppressed) | V1 remote confusion not reproduced |
| empty_desk | Empty desk | **12** (bottle:7, person:4, book:1) | 0 | toothbrush/tie NOT reproduced |

---

## Phase 4 Results

**Status: ALL 20 SCENARIOS MEASURED**
Evaluation date: 2026-09-02. All numbers sourced directly from
`data/evaluation/phase4_results.csv` (3,223 rows total).

---

### Overall raw counts (all 20 scenarios combined)

| Observation type | Count |
|---|---|
| correct_class_observation | 91 |
| wrong_class_observation | 325 |
| no_detection_observation | 348 |
| false_positive_observation | 1,049 |
| ignored_class_suppressed | 54 |
| gap_observation (Cat 6) | 182 |
| empty_scene_correct | 1,174 |
| **Total rows** | **3,223** |

---

### Category 1 — True Positive Validation Results

| Scenario | Frames | Correct-class | Wrong-class | No-det | Accepted classes | Conf mean |
|---|---|---|---|---|---|---|
| person_standing | 30 | **30** | 0 | 0 | person:30 | 0.921 |
| chair_visible | 30 | 0 | **30** | 0 | person:30 | 0.921 |
| couch_visible | 30 | 0 | **30** | 0 | person:30 | 0.920 |
| dining_table_visible | 30 | 0 | 29 | **47** | person:29 | 0.736 |
| laptop_visible | 30 | 0 | **59** | 8 | person:31, bottle:28 | 0.760 |
| bottle_visible | 30 | 9 | **35** | 65 | person:35, bottle:9 | 0.586 |
| backpack_visible | 30 | 0 | **41** | 24 | person:31, chair:10 | 0.607 |

**Key finding — Category 1:**
Only `person` was detected correctly and consistently (30/30 frames, conf mean 0.921).
All other navigation-critical objects were **never detected as their correct class** or
detected only rarely. The dominant wrong-class detection across all scenarios was
`person` — the model interpreted chairs, couches, tables, laptops, bottles, and
backpacks as people in a large proportion of frames.

---

### Category 2 — False Positive Probe Results

| Scenario | Frames | FP obs (accepted) | FP/min | Accepted classes |
|---|---|---|---|---|
| usb_charger | 30 | 65 | 2,774 | person:42, cell_phone:23 |
| pen_or_pencil | 30 | 74 | 3,108 | person:32, laptop:20, book:14, cell_phone:8 |
| phone_only | 30 | 0 | 0.00 | person:3, cell_phone:1 |
| empty_desk | 30 | 12 | 517 | bottle:7, person:4, book:1 |

**Key finding — Category 2:**
- **USB charger**: produced 65 accepted FP detections in 30 frames. The V1 concern
  (charger → cell phone) was confirmed: 23 `cell phone` detections. Additionally,
  42 `person` detections were produced — the background/user reflection was the
  likely cause.
- **Pen/pencil**: produced 74 accepted FP detections. The V1 concern (pen → baseball
  bat) was NOT observed — `baseball bat` did not appear. However the pen triggered
  `laptop` (20), `book` (14), and `cell phone` (8) misclassifications. Person was
  also detected (32), again likely from background.
- **Phone**: largely undetected — only 4 accepted detections in 30 frames (1 correct
  `cell phone`, 3 wrong `person`). The V1 concern (phone → remote) was NOT confirmed;
  `remote` was correctly suppressed (5 ignored events).
- **Empty desk**: 12 FP observations. No toothbrush or tie appeared (V1 phantoms not
  reproduced). Bottle (7) and person (4) were falsely accepted.

---

### Category 3 — Empty Scene FP Rate (60-second test)

| Scene | Duration (s) | Frames | Accepted total | Accepted/min |
|---|---|---|---|---|
| empty_scene_fp_rate | 60.01 | 742 | 898 | **897.81** |

Accepted class breakdown over 60 seconds:

| Class | Count |
|---|---|
| person | 612 |
| cell phone | 145 |
| chair | 78 |
| bottle | 50 |
| laptop | 11 |
| book | 2 |

**Key finding — Category 3:**
The empty-scene FP rate of ~898 accepted detections/minute is very high. This
is the single most important finding of Phase 4. It means that in the tested
environment (a room with background clutter), YOLO11n continuously produces
navigation-relevant detections even when no navigation objects are intentionally
present. `person` dominated at 612 detections — the background was being
misclassified as a person at high frequency.

This result directly informs the Phase 10 temporal confirmation parameters:
`CONFIRMATION_FRAMES` and `CLASS_STABILITY_THRESHOLD` must be set to filter out
the overwhelming majority of these per-frame FPs before they reach the alert system.

---

### Category 4 — Partial Visibility / Occlusion Results

| Scenario | Frames | Correct-class | Wrong-class | No-det | Accepted classes | Conf mean |
|---|---|---|---|---|---|---|
| partial_person | 30 | 32 | 2 | 36 | person:32, book:1, laptop:1 | 0.721 |
| partial_chair | 30 | 0 | 22 | 28 | cell_phone:17, person:5 | 0.625 |
| edge_of_frame_bottle | 30 | 0 | 47 | 23 | cell_phone:28, person:19 | 0.533 |

**Key finding — Category 4:**
- Partially occluded person: still detected correctly in 32/30 row observations
  (multiple detections per frame counted). Confidence dropped to mean 0.721 vs
  0.921 fully visible — a meaningful reduction under occlusion.
- Partially occluded chair: never detected as `chair`. `cell phone` dominated (17).
- Bottle at frame edge: never detected as `bottle`. `cell phone` dominated (28).
  Rectangular objects at the frame edge are consistently misclassified as
  `cell phone` — a systematic pattern worth noting.

---

### Category 5 — Multiple Objects Results

| Scenario | Expected | Frames | Correct-class | Wrong-class | No-det | Accepted classes |
|---|---|---|---|---|---|---|
| person_and_chair | person+chair | 30 | 8 | 3 | 34 | person:8, cell_phone:3 |
| bottle_laptop_backpack | bottle+laptop+backpack | 30 | 11 | 24 | 27 | cell_phone:14, laptop:11, person:9, chair:1 |

**Key finding — Category 5:**
- Person+Chair: `chair` was never detected. Person was detected in only 8/30 row
  observations. Confidence was low (mean 0.431) suggesting the multi-object scene
  confused the model.
- Bottle+Laptop+Backpack: `laptop` was detected correctly (11 times). `bottle` and
  `backpack` were never detected as their correct class. `cell phone` (14) dominated
  the wrong-class detections — rectangular screen-like objects strongly trigger it.

---

### Category 6 — COCO Gap Analysis Results

| Scenario | Frames | Gap obs. | Accepted classes (misclassifications) |
|---|---|---|---|
| door_gap_analysis | 30 | 76 | couch:13, person:11, chair:9, cell_phone:2 |
| doorway_gap_analysis | 30 | 52 | couch:5, chair:3, person:1 |
| stairs_gap_analysis | 30 | 54 | couch:29, chair:4, person:1 |

**Key finding — Category 6:**
Doors and stairs produced **zero correct detections** as expected — they are not
COCO classes. However the model did NOT produce zero detections. Instead it
consistently misclassified these scenes:

- **Door** → misclassified as `couch` (13), `person` (11), `chair` (9)
- **Doorway** → misclassified as `couch` (5), `chair` (3)
- **Stairs** → misclassified as `couch` (29) at confidence mean 0.689

The `couch` misclassification for stairs is particularly notable — it was consistent
(29/34 accepted detections) and at moderate confidence (0.689). This is a hazardous
pattern: the system might say "couch ahead" when stairs are present. This gap must
be documented prominently in the safety disclaimer.

**These misclassifications would reach the user as wrong alerts if no mitigation
is applied.** They are not filtered by the ignored list because `couch` and `chair`
are valid navigation classes. Temporal confirmation (Phase 10) and proximity-based
gating are the primary mitigations in the current design.

---

### Ignored-class suppression summary

The Phase 3 filter's ignored list correctly blocked these V1 problem classes
from reaching accepted navigation output:

| Suppressed class | Count blocked |
|---|---|
| toothbrush | 38 |
| tie | 12 |
| knife | 2 |
| remote | 2 |
| **Total** | **54** |

These 54 detections would have reached the alert system in V1. The V2 filter
correctly suppressed all of them. However `baseball bat` did not appear in any
scenario — the pen/pencil V1 confusion manifested differently (as `laptop`/`book`
instead) under YOLO11n.

---

### Overall findings and implications for Phase 5+

**What works well:**
1. `person` detection is reliable and consistent when a person is clearly
   visible (30/30, conf 0.921). This is the most safety-critical class.
2. The ignored-class filter correctly suppresses all V1 phantom classes.
3. Blank frames produce zero detections — no hallucination on null input.

**What does not work well:**
1. Non-person navigation objects are detected poorly or not at all. Chair,
   couch, table, backpack, and bottle detection is unreliable on this hardware
   and in this environment.
2. The `person` class dominates wrong-class detections across all scenarios —
   the model conflates many objects and background elements with people.
3. The empty-scene FP rate (~898/min) is far too high to use raw detection
   output for navigation alerts without temporal filtering.
4. Rectangular objects at frame edges → `cell phone` is a systematic pattern.
5. Stairs → `couch` (29/34 frames) is a safety-critical misclassification gap.

**Direct implications for subsequent phases:**
- **Phase 5 (Tracking):** Track stability will be poor for non-person classes
  due to inconsistent detections. Design tracking parameters conservatively.
- **Phase 10 (Temporal confirmation):** `CONFIRMATION_FRAMES` must be high
  enough to absorb the ~898 FP/min raw rate. 5 frames is likely insufficient;
  8–10 frames should be evaluated.
- **Phase 15 (FP reduction):** The `person` over-detection pattern needs
  dedicated investigation — possibly minimum bounding-box size filtering.
- **Future:** Open-vocabulary detection or a secondary classifier for doors/stairs
  should be evaluated to close the COCO gap for safety-critical obstacles.

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
