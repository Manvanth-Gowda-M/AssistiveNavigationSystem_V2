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
| door_gap_analysis | NOT YET MEASURED | — | — | — |
| doorway_gap_analysis | NOT YET MEASURED | — | — | — |
| stairs_gap_analysis | NOT YET MEASURED | — | — | — |

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
| usb_charger | USB charger | NOT YET MEASURED | NOT YET MEASURED | V1: charger→phone |
| pen_or_pencil | Pen/pencil | NOT YET MEASURED | NOT YET MEASURED | V1: pen→baseball bat |
| phone_only | Mobile phone | NOT YET MEASURED | NOT YET MEASURED | V1: phone→remote |
| empty_desk | Empty desk | NOT YET MEASURED | NOT YET MEASURED | V1: phantom detections |

---

## Phase 4 Results

> **Status: NOT YET MEASURED**
>
> All result fields below will be populated after the user runs the
> interactive evaluation (`phase4_eval.py`). Nothing is pre-filled.
>
> Do not interpret the "NOT YET MEASURED" placeholders as results.

### Category 1 — True Positive Validation Results

| Scenario | Frames | Correct-class obs. | Wrong-class obs. | No-det obs. | Conf mean |
|---|---|---|---|---|---|
| person_standing | NM | NM | NM | NM | NM |
| chair_visible | NM | NM | NM | NM | NM |
| couch_visible | NM | NM | NM | NM | NM |
| dining_table_visible | NM | NM | NM | NM | NM |
| laptop_visible | NM | NM | NM | NM | NM |
| bottle_visible | NM | NM | NM | NM | NM |
| backpack_visible | NM | NM | NM | NM | NM |

NM = Not yet measured

### Category 3 — Empty Scene FP Rate

| Scene | Duration (s) | Frames | Accepted total | Accepted/min |
|---|---|---|---|---|
| empty_scene_fp_rate | NM | NM | NM | NM |

### Overall findings

> To be completed after all scenarios are run.

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
