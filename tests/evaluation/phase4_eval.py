"""
Phase 4 Detection Evaluation — Interactive Runner
===================================================
Runs structured evaluation scenarios against YOLO11n to measure how the
detector actually behaves on this computer, webcam, and room environment.

This tool MEASURES. It does NOT claim or invent results.

Scenarios:
  Category 1 — True Positive Validation (navigation objects)
  Category 2 — False Positive Probe    (V1 problem objects)
  Category 3 — Empty Scene FP Rate
  Category 4 — Partial Visibility / Occlusion
  Category 5 — Multiple Objects Together
  Category 6 — Doors and Stairs Gap Analysis

Usage:
    python tests/evaluation/phase4_eval.py
    python tests/evaluation/phase4_eval.py --frames 60
    python tests/evaluation/phase4_eval.py --fps-duration 120
    python tests/evaluation/phase4_eval.py --no-display
    python tests/evaluation/phase4_eval.py --categories 1,2,3
    python tests/evaluation/phase4_eval.py --help

Output:
    data/evaluation/phase4_results.csv  — per-frame raw detection data
    data/evaluation/phase4_report.txt   — human-readable report

IMPORTANT:
    This script requires your physical presence at the camera.
    Before each scenario, you will be told what to place in the frame.
    Results are written to CSV and report files ONLY from actual camera data.
    Nothing is pre-populated or fabricated.

SAFETY NOTE:
    This is a research prototype evaluation.
    Detection results do not imply the system is suitable for
    independent navigation or a replacement for a mobility aid.
"""

import sys
import csv
import time
import argparse
import textwrap
import statistics
import platform
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

# Ensure src/ is importable when run from project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import cv2
import numpy as np

from assistive_navigation.utils.config_loader import load_config, get_logs_dir
from assistive_navigation.utils.logger import get_logger
from assistive_navigation.camera.capture import CameraCapture, CameraError
from assistive_navigation.detection.detector import (
    ObjectDetector, Detection, DetectionResult, DetectorError
)
from assistive_navigation.detection.filter import (
    DetectionFilter, FilterResult, Priority, RejectionReason
)

logger = get_logger(__name__, level="INFO")

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result codes
# Each frame in a scenario is categorised using one of these strings.
# Conservative language: we call them "observations", not precision/recall.
# ─────────────────────────────────────────────────────────────────────────────
class FrameResult:
    CORRECT_CLASS    = "correct_class_observation"   # expected class accepted
    WRONG_CLASS      = "wrong_class_observation"     # different class accepted
    NO_DETECTION     = "no_detection_observation"    # nothing accepted
    FP_OBSERVATION   = "false_positive_observation"  # unexpected accepted detection
    EMPTY_CORRECT    = "empty_scene_correct"         # expected: nothing; got: nothing
    IGNORED_SUPPRESSED = "ignored_class_suppressed"  # V1 problem class correctly blocked


# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────────
CATEGORY_1_SCENARIOS = [
    {
        "id": "person_standing",
        "category": "1_true_positive",
        "expected_class": "person",
        "display_name": "Person standing",
        "instruction": (
            "Stand or have someone stand in front of the camera.\n"
            "Stay still for the duration of the test.\n"
            "You should be clearly visible and centred in the frame."
        ),
    },
    {
        "id": "chair_visible",
        "category": "1_true_positive",
        "expected_class": "chair",
        "display_name": "Chair",
        "instruction": (
            "Place a chair clearly in view of the camera.\n"
            "The full or most of the chair should be in frame."
        ),
    },
    {
        "id": "couch_visible",
        "category": "1_true_positive",
        "expected_class": "couch",
        "display_name": "Couch / sofa",
        "available": False,
        "not_tested_reason": (
            "No real couch or sofa was available for physical testing. "
            "This scenario must not be run with a phone image or any "
            "substitute. Marked NOT TESTED until a real couch is available."
        ),
        "instruction": (
            "Point the camera at a couch or sofa.\n"
            "A significant portion of it should fill the frame.\n"
            "NOTE: This scenario is currently marked NOT TESTED because\n"
            "no real couch is available."
        ),
    },
    {
        "id": "dining_table_visible",
        "category": "1_true_positive",
        "expected_class": "dining table",
        "display_name": "Dining table",
        "instruction": (
            "Point the camera at a table (dining or similar).\n"
            "The table surface/edge should be clearly visible."
        ),
    },
    {
        "id": "laptop_visible",
        "category": "1_true_positive",
        "expected_class": "laptop",
        "display_name": "Laptop",
        "instruction": (
            "Place an open laptop on a surface in the camera view.\n"
            "The screen and keyboard should be visible."
        ),
    },
    {
        "id": "bottle_visible",
        "category": "1_true_positive",
        "expected_class": "bottle",
        "display_name": "Bottle",
        "instruction": (
            "Place a bottle (water, drink, etc.) in clear view.\n"
            "It should be upright and clearly visible."
        ),
    },
    {
        "id": "backpack_visible",
        "category": "1_true_positive",
        "expected_class": "backpack",
        "display_name": "Backpack",
        "instruction": (
            "Place a backpack in clear view of the camera.\n"
            "It can be on the floor or on a surface."
        ),
    },
]

CATEGORY_2_SCENARIOS = [
    {
        "id": "usb_charger",
        "category": "2_false_positive_probe",
        "expected_class": "no_navigation_object",
        "display_name": "USB charger (plug + cable)",
        "v1_concern": "Previously observed as: cell phone",
        "instruction": (
            "Place a USB charger (plug + cable or just the plug) on a surface.\n"
            "Centre it in the camera view.\n"
            "This tests whether the charger is misclassified as a cell phone."
        ),
    },
    {
        "id": "pen_or_pencil",
        "category": "2_false_positive_probe",
        "expected_class": "no_navigation_object",
        "display_name": "Pen or pencil",
        "v1_concern": "Previously observed as: baseball bat",
        "instruction": (
            "Hold a pen or pencil upright or place it on a surface.\n"
            "Centre it in the camera view.\n"
            "This tests whether the pen is misclassified as a baseball bat."
        ),
    },
    {
        "id": "phone_only",
        "category": "2_false_positive_probe",
        "expected_class": "cell phone",   # phone IS in the navigation list (contextual)
        "display_name": "Mobile phone",
        "v1_concern": "Previously observed as: remote",
        "instruction": (
            "Place a mobile phone flat or upright in clear camera view.\n"
            "This tests whether the phone is correctly identified as 'cell phone'\n"
            "or misclassified as 'remote'."
        ),
    },
    {
        "id": "empty_desk",
        "category": "2_false_positive_probe",
        "expected_class": "no_navigation_object",
        "display_name": "Empty desk (no navigation objects)",
        "v1_concern": "Previously observed: toothbrush, tie, misc phantom detections",
        "instruction": (
            "Clear your desk of any navigation objects.\n"
            "Point the camera at the empty desk surface.\n"
            "This tests for phantom/hallucinated detections."
        ),
    },
]

CATEGORY_3_SCENARIOS = [
    {
        "id": "empty_scene_fp_rate",
        "category": "3_empty_scene",
        "expected_class": "no_navigation_object",
        "display_name": "Empty scene — false positive rate measurement",
        "instruction": (
            "Point the camera at a scene with NO navigation-relevant objects:\n"
            "  - empty wall, empty floor, or cluttered background\n"
            "  - remove any person, chair, bottle, backpack, or other\n"
            "    navigation objects from the camera view\n"
            "The camera will run for the configured duration.\n"
            "Any accepted detection is an observed false positive for this scene."
        ),
    },
]

CATEGORY_4_SCENARIOS = [
    {
        "id": "partial_person",
        "category": "4_occlusion",
        "expected_class": "person",
        "display_name": "Person — partially occluded (~50%)",
        "instruction": (
            "Stand in front of the camera, then step half behind a wall,\n"
            "door frame, or large object so approximately half of you\n"
            "is hidden. Keep the other half clearly visible."
        ),
    },
    {
        "id": "partial_chair",
        "category": "4_occlusion",
        "expected_class": "chair",
        "display_name": "Chair — partially occluded (~50%)",
        "instruction": (
            "Place a chair so only about half of it is visible.\n"
            "The other half should be behind a wall, door, or object.\n"
            "Or place the chair at the very edge of the frame."
        ),
    },
    {
        "id": "edge_of_frame_bottle",
        "category": "4_occlusion",
        "expected_class": "bottle",
        "display_name": "Bottle — at left or right edge of frame",
        "instruction": (
            "Place a bottle so it is at the far left or right edge of\n"
            "the camera view — only about half the bottle should be visible."
        ),
    },
]

CATEGORY_5_SCENARIOS = [
    {
        "id": "person_and_chair",
        "category": "5_multiple_objects",
        "expected_class": "multiple",
        "expected_classes": ["person", "chair"],
        "display_name": "Person AND Chair together",
        "instruction": (
            "Stand next to a chair so both are clearly visible in the\n"
            "same camera frame at the same time."
        ),
    },
    {
        "id": "bottle_laptop_backpack",
        "category": "5_multiple_objects",
        "expected_class": "multiple",
        "expected_classes": ["bottle", "laptop", "backpack"],
        "display_name": "Bottle + Laptop + Backpack",
        "instruction": (
            "Place a bottle, an open laptop, and a backpack on a surface\n"
            "so all three are visible in the same camera frame."
        ),
    },
]

CATEGORY_6_SCENARIOS = [
    {
        "id": "door_gap_analysis",
        "category": "6_gap_analysis",
        "expected_class": "COCO_GAP",  # not a COCO class
        "display_name": "Door — COCO capability gap test",
        "coco_note": (
            "YOLO11n was trained on COCO 80 classes. "
            "'door' is NOT one of them."
        ),
        "instruction": (
            "Point the camera at a closed door.\n"
            "The door should fill a significant part of the frame.\n"
            "We are recording what YOLO11n actually detects (if anything).\n"
            "There is no 'correct' detection — this measures the gap."
        ),
    },
    {
        "id": "doorway_gap_analysis",
        "category": "6_gap_analysis",
        "expected_class": "COCO_GAP",
        "display_name": "Doorway (open door frame) — COCO capability gap test",
        "coco_note": (
            "YOLO11n was trained on COCO 80 classes. "
            "'doorway' is NOT one of them."
        ),
        "instruction": (
            "Point the camera at an open door frame / doorway.\n"
            "Record what YOLO11n detects through or around the doorway."
        ),
    },
    {
        "id": "stairs_gap_analysis",
        "category": "6_gap_analysis",
        "expected_class": "COCO_GAP",
        "display_name": "Stairs / staircase — COCO capability gap test",
        "coco_note": (
            "YOLO11n was trained on COCO 80 classes. "
            "'stairs' is NOT one of them."
        ),
        "instruction": (
            "Point the camera at a staircase or set of steps.\n"
            "This is a safety-critical gap test.\n"
            "Record what YOLO11n detects (if anything) when viewing stairs."
        ),
    },
]

ALL_SCENARIOS = (
    CATEGORY_1_SCENARIOS
    + CATEGORY_2_SCENARIOS
    + CATEGORY_3_SCENARIOS
    + CATEGORY_4_SCENARIOS
    + CATEGORY_5_SCENARIOS
    + CATEGORY_6_SCENARIOS
)


# ─────────────────────────────────────────────────────────────────────────────
# Per-frame evaluation logic
# This module is independently testable — no camera or model required.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_frame(
    scenario: dict,
    filtered: FilterResult,
    det_result: DetectionResult,
) -> List[dict]:
    """
    Evaluate one frame against a scenario's ground truth.

    Returns a list of row dicts ready to write to the results CSV.
    One row per detection (raw), plus one row if there are zero detections.

    This function never touches the camera or model — it only interprets
    the data structures from Phase 3.

    Args:
        scenario:   Scenario definition dict (from lists above).
        filtered:   FilterResult from DetectionFilter.filter().
        det_result: DetectionResult from ObjectDetector.detect().

    Returns:
        List[dict] — each dict has keys matching the CSV columns.
    """
    rows = []
    expected = scenario.get("expected_class", "unknown")
    category = scenario.get("category", "unknown")
    ts = datetime.now().isoformat(timespec="milliseconds")

    # Build a fast lookup: which detections were accepted?
    accepted_ids = {id(d) for d in filtered.accepted}

    # ── Case A: no raw detections at all ────────────────────────────────────
    if det_result.raw_count == 0:
        if expected == "no_navigation_object" or expected == "COCO_GAP":
            result_code = FrameResult.EMPTY_CORRECT
        else:
            result_code = FrameResult.NO_DETECTION

        rows.append({
            "scenario":          scenario["id"],
            "category":          category,
            "timestamp":         ts,
            "ground_truth":      expected,
            "detected_class":    "",
            "confidence":        "",
            "bbox_x1":           "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
            "accepted":          False,
            "eval_result":       result_code,
            "rejection_reason":  "",
            "raw_latency_ms":    f"{det_result.latency_ms:.2f}",
        })
        return rows

    # ── Case B: process each raw detection ──────────────────────────────────
    # Check if any accepted detection matches expected class
    accepted_classes = [d.class_name for d in filtered.accepted]
    expected_classes = scenario.get("expected_classes", [expected])

    for det in det_result.detections:
        is_accepted = id(det) in accepted_ids

        # Find rejection reason if rejected
        rej_reason = ""
        if not is_accepted:
            for r in filtered.rejected:
                if (r["class_id"] == det.class_id and
                        abs(r["confidence"] - det.confidence) < 0.001):
                    rej_reason = r["reason"]
                    break

        # Determine evaluation result for this detection
        if expected == "COCO_GAP":
            # Gap analysis: simply record what was detected — no pass/fail
            result_code = "gap_observation"

        elif expected == "no_navigation_object":
            # Empty-scene / FP-probe scenario
            if is_accepted:
                result_code = FrameResult.FP_OBSERVATION
            elif rej_reason == RejectionReason.CLASS_IGNORED:
                result_code = FrameResult.IGNORED_SUPPRESSED
            else:
                result_code = FrameResult.EMPTY_CORRECT

        elif expected == "multiple":
            # Multi-object scenario
            if is_accepted and det.class_name in expected_classes:
                result_code = FrameResult.CORRECT_CLASS
            elif is_accepted and det.class_name not in expected_classes:
                result_code = FrameResult.WRONG_CLASS
            elif not is_accepted:
                result_code = FrameResult.NO_DETECTION
            else:
                result_code = FrameResult.WRONG_CLASS

        else:
            # Single-object scenario
            if is_accepted and det.class_name == expected:
                result_code = FrameResult.CORRECT_CLASS
            elif is_accepted and det.class_name != expected:
                result_code = FrameResult.WRONG_CLASS
            elif not is_accepted and rej_reason == RejectionReason.CLASS_IGNORED:
                result_code = FrameResult.IGNORED_SUPPRESSED
            else:
                # Detected something but it was rejected (low conf or not in list)
                result_code = FrameResult.NO_DETECTION

        x1, y1, x2, y2 = det.bbox
        rows.append({
            "scenario":          scenario["id"],
            "category":          category,
            "timestamp":         ts,
            "ground_truth":      expected,
            "detected_class":    det.class_name,
            "confidence":        f"{det.confidence:.4f}",
            "bbox_x1":           f"{x1:.1f}",
            "bbox_y1":           f"{y1:.1f}",
            "bbox_x2":           f"{x2:.1f}",
            "bbox_y2":           f"{y2:.1f}",
            "accepted":          is_accepted,
            "eval_result":       result_code,
            "rejection_reason":  rej_reason,
            "raw_latency_ms":    f"{det_result.latency_ms:.2f}",
        })

    return rows


def compute_scenario_summary(
    scenario: dict,
    rows: List[dict],
    n_frames: int,
    duration_s: float,
) -> dict:
    """
    Compute summary statistics for one completed scenario.

    Uses conservative terminology throughout — no precision/recall/mAP.

    Args:
        scenario:  Scenario definition dict.
        rows:      All CSV rows accumulated during this scenario.
        n_frames:  Total frames processed.
        duration_s: Wall-clock seconds elapsed.

    Returns:
        dict with summary statistics.
    """
    total_rows = len(rows)
    expected   = scenario.get("expected_class", "unknown")

    # Count by eval_result code
    result_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        result_counts[r["eval_result"]] += 1

    # Confidence statistics for accepted detections
    accepted_confs = [
        float(r["confidence"])
        for r in rows
        if r["accepted"] is True and r["confidence"] != ""
    ]

    conf_stats = {}
    if accepted_confs:
        conf_stats = {
            "conf_min":  min(accepted_confs),
            "conf_max":  max(accepted_confs),
            "conf_mean": statistics.mean(accepted_confs),
            "conf_stdev": statistics.stdev(accepted_confs) if len(accepted_confs) > 1 else 0.0,
        }

    # Accepted class distribution
    class_counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r["accepted"] and r["detected_class"]:
            class_counts[r["detected_class"]] += 1

    # Frames with at least one correct-class observation
    # (frame_number not tracked per-row here; rows may be multiple per frame)
    correct_frames   = result_counts.get(FrameResult.CORRECT_CLASS, 0)
    wrong_frames     = result_counts.get(FrameResult.WRONG_CLASS, 0)
    no_det_frames    = result_counts.get(FrameResult.NO_DETECTION, 0)
    fp_events        = result_counts.get(FrameResult.FP_OBSERVATION, 0)
    suppressed_events= result_counts.get(FrameResult.IGNORED_SUPPRESSED, 0)
    gap_obs          = result_counts.get("gap_observation", 0)
    empty_correct    = result_counts.get(FrameResult.EMPTY_CORRECT, 0)

    fp_per_min = (fp_events / duration_s * 60.0) if duration_s > 0 else 0.0

    return {
        "scenario_id":        scenario["id"],
        "category":           scenario["category"],
        "expected_class":     expected,
        "n_frames":           n_frames,
        "duration_s":         round(duration_s, 2),
        "total_raw_detections": total_rows,
        "correct_class_observations": correct_frames,
        "wrong_class_observations":   wrong_frames,
        "no_detection_observations":  no_det_frames,
        "fp_observations":            fp_events,
        "ignored_suppressed_events":  suppressed_events,
        "gap_observations":           gap_obs,
        "empty_correct":              empty_correct,
        "accepted_class_counts":      dict(class_counts),
        "fp_per_minute":              round(fp_per_min, 2),
        **conf_stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV overlay for live display
# ─────────────────────────────────────────────────────────────────────────────
def draw_eval_overlay(
    frame: np.ndarray,
    scenario: dict,
    filtered: FilterResult,
    det_result: DetectionResult,
    frame_num: int,
    total_frames: int,
) -> np.ndarray:
    """
    Draw evaluation HUD on a frame copy.
    Does not modify the original frame.
    """
    overlay = frame.copy()
    h, w = overlay.shape[:2]

    # Accepted detections — green box
    for det in filtered.accepted:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(overlay, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

    # Rejected detections — dim red box
    for r in filtered.rejected:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        short = {
            RejectionReason.LOW_CONFIDENCE:    "low_conf",
            RejectionReason.CLASS_IGNORED:     "ignored",
            RejectionReason.CLASS_NOT_IN_LIST: "not_in_list",
        }.get(r["reason"], r["reason"])
        label = f"{r['class_name']} {r['confidence']:.2f} [{short}]"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 60, 180), 1)
        cv2.putText(overlay, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 200), 1)

    # HUD
    hud_lines = [
        f"Scenario: {scenario['id']}  [{scenario['category']}]",
        f"Expected: {scenario.get('expected_class','?')}",
        f"Frame {frame_num}/{total_frames}  "
        f"Acc:{filtered.accepted_count} Rej:{filtered.rejected_count}  "
        f"{det_result.latency_ms:.0f}ms",
    ]
    for i, line in enumerate(hud_lines):
        cv2.putText(overlay, line, (8, 20 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)

    # Bottom instruction bar
    cv2.putText(overlay,
                "GREEN=accepted  RED=rejected  Q=skip scenario",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────
def build_report(
    summaries: List[dict],
    config: dict,
    output_path: Path,
    skipped_scenarios: List[str],
    not_tested_ids: set = None,
    cli_skipped_ids: set = None,
) -> None:
    """
    Write a human-readable evaluation report.

    Reports ONLY what was actually measured.
    Distinguishes three non-measured states:
      NOT TESTED   : scenario marked available=False (e.g. couch — no physical object)
      SKIPPED      : user pressed S during the run, OR in --skip-scenarios
      NOT YET MEASURED : scenario was in scope but not reached

    Args:
        summaries:          List of summary dicts from compute_scenario_summary().
        config:             Full config dict.
        output_path:        Where to write the .txt report.
        skipped_scenarios:  IDs skipped by user pressing S during the run.
        not_tested_ids:     IDs of scenarios marked available=False.
        cli_skipped_ids:    IDs passed via --skip-scenarios CLI argument.
    """
    not_tested_ids   = not_tested_ids   or set()
    cli_skipped_ids  = cli_skipped_ids  or set()
    # All skipped-by-user or by CLI: treated as user choice, not unavailability
    all_user_skipped = set(skipped_scenarios) | cli_skipped_ids
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summ_by_id = {s["scenario_id"]: s for s in summaries}

    lines = []
    W = 70  # line width

    def sep(char="="):
        lines.append(char * W)

    def heading(text):
        sep()
        lines.append(f"  {text}")
        sep()

    def sub(text):
        lines.append(f"\n  -- {text} --")

    def row(label, value, indent=4):
        lines.append(f"{' ' * indent}{label:<38}: {value}")

    heading("PHASE 4 DETECTION EVALUATION REPORT")
    lines.append(f"    Generated : {now}")
    lines.append(f"    Platform  : {platform.system()} {platform.release()}")
    lines.append(f"    Python    : {platform.python_version()}")
    lines.append(f"    Model     : {config['detection']['model_name']}")
    lines.append(f"    Conf threshold : {config['detection']['confidence_threshold']}")
    lines.append(f"    Device    : {config['detection']['device']}")
    lines.append("")
    lines.append("  IMPORTANT: All results below describe the observed behavior")
    lines.append("  of YOLO11n on this specific camera, room, and lighting at")
    lines.append("  the time of evaluation. They do NOT constitute a formal")
    lines.append("  accuracy benchmark and cannot be generalised to other")
    lines.append("  environments or hardware.")
    lines.append("")
    lines.append("  This system is a RESEARCH PROTOTYPE. It must not be used")
    lines.append("  as a substitute for a mobility aid or human assistance.")

    # ── Per-scenario results ─────────────────────────────────────────────────
    def _print_scenario(sc):
        sid = sc["id"]
        heading(f"SCENARIO: {sc['display_name']}")
        lines.append(f"    ID       : {sid}")
        lines.append(f"    Category : {sc['category']}")
        lines.append(f"    Expected : {sc.get('expected_class', '?')}")

        if sc.get("v1_concern"):
            lines.append(f"    V1 concern: {sc['v1_concern']}")
        if sc.get("coco_note"):
            lines.append(f"    COCO note : {sc['coco_note']}")

        if sid in not_tested_ids:
            # Scenario explicitly marked unavailable — not a user skip
            lines.append("")
            sc_obj = next((s for s in ALL_SCENARIOS if s["id"] == sid), None)
            reason = sc_obj.get("not_tested_reason", "No reason given.") if sc_obj else "No reason given."
            lines.append("    STATUS: NOT TESTED")
            lines.append(f"    Reason: {reason}")
            return

        if sid in all_user_skipped:
            lines.append("")
            lines.append("    STATUS: SKIPPED BY USER")
            return

        s = summ_by_id.get(sid)
        if s is None:
            lines.append("")
            lines.append("    STATUS: NOT YET MEASURED")
            lines.append("    Run phase4_eval.py to collect results.")
            return

        lines.append(f"    STATUS: MEASURED")
        lines.append("")
        row("Frames processed",         s["n_frames"])
        row("Duration (seconds)",        s["duration_s"])
        row("Total raw detections",      s["total_raw_detections"])
        sep("-")
        row("Correct-class observations",  s["correct_class_observations"])
        row("Wrong-class observations",    s["wrong_class_observations"])
        row("No-detection observations",   s["no_detection_observations"])
        row("FP observations (accepted)",  s["fp_observations"])
        row("Ignored-suppressed events",   s["ignored_suppressed_events"])
        if s["gap_observations"]:
            row("Gap-analysis observations", s["gap_observations"])
        row("FP rate (accepted/min)",     f"{s['fp_per_minute']:.2f}")
        sep("-")

        if s["accepted_class_counts"]:
            lines.append("    Accepted class distribution:")
            for cls, cnt in sorted(s["accepted_class_counts"].items(),
                                   key=lambda x: -x[1]):
                lines.append(f"        {cls:<25}: {cnt}")

        if "conf_mean" in s:
            lines.append("    Confidence (accepted detections):")
            row("  mean",   f"{s['conf_mean']:.3f}")
            row("  min",    f"{s['conf_min']:.3f}")
            row("  max",    f"{s['conf_max']:.3f}")
            row("  stdev",  f"{s['conf_stdev']:.3f}")

    # Print all scenarios in order
    for sc in ALL_SCENARIOS:
        _print_scenario(sc)

    # ── Overall summary ──────────────────────────────────────────────────────
    heading("OVERALL FINDINGS")
    measured = [s for s in summaries]
    not_measured = [
        sc["id"] for sc in ALL_SCENARIOS
        if (sc["id"] not in summ_by_id
            and sc["id"] not in all_user_skipped
            and sc["id"] not in not_tested_ids)
    ]

    lines.append(f"    Scenarios measured     : {len(measured)}")
    lines.append(f"    NOT TESTED (unavailable): {len(not_tested_ids)}"
                 + (f" ({', '.join(not_tested_ids)})" if not_tested_ids else ""))
    lines.append(f"    Skipped (user choice)  : {len(all_user_skipped)}")
    lines.append(f"    Not yet measured       : {len(not_measured)}")
    if not_measured:
        lines.append(f"    IDs not yet measured   : {', '.join(not_measured)}")

    if measured:
        total_fp = sum(s["fp_observations"] for s in measured)
        total_correct = sum(s["correct_class_observations"] for s in measured)
        total_wrong = sum(s["wrong_class_observations"] for s in measured)
        total_no_det = sum(s["no_detection_observations"] for s in measured)
        total_suppressed = sum(s["ignored_suppressed_events"] for s in measured)
        lines.append("")
        lines.append("    Aggregated across all MEASURED scenarios:")
        row("Total correct-class observations", total_correct)
        row("Total wrong-class observations",   total_wrong)
        row("Total no-detection observations",  total_no_det)
        row("Total FP observations (accepted)", total_fp)
        row("Total ignored-class suppressions", total_suppressed)

    # ── COCO gap section ─────────────────────────────────────────────────────
    heading("YOLO11n CAPABILITY GAP: DOORS AND STAIRS")
    lines.append("  YOLO11n is trained on COCO 80 classes.")
    lines.append("  Neither 'door', 'stairs', 'doorway', nor 'step'")
    lines.append("  is a COCO class.")
    lines.append("")
    lines.append("  This means YOLO11n cannot reliably detect doors or stairs.")
    lines.append("  It may produce:")
    lines.append("    - Zero detections  (most likely)")
    lines.append("    - Misclassified detections (e.g. door as 'refrigerator' or 'tv')")
    lines.append("    - Very low confidence incidental detections")
    lines.append("")
    lines.append("  DO NOT interpret a non-zero detection as reliable door/stair detection.")
    lines.append("  This is a documented architectural limitation of YOLO11n for this use case.")
    lines.append("")
    lines.append("  Observed results from Category 6 scenarios:")

    for sc in CATEGORY_6_SCENARIOS:
        sid = sc["id"]
        s = summ_by_id.get(sid)
        lines.append(f"\n    {sc['display_name']}:")
        if sid in skipped_scenarios:
            lines.append("      SKIPPED")
        elif s is None:
            lines.append("      NOT YET MEASURED")
        else:
            if s["accepted_class_counts"]:
                lines.append(f"      Accepted detections: {dict(s['accepted_class_counts'])}")
            else:
                lines.append(f"      Accepted detections: NONE")
            lines.append(f"      No-detection observations: {s['no_detection_observations']}")
            lines.append(f"      Total frames: {s['n_frames']}")

    # ── V1 FP probe section ──────────────────────────────────────────────────
    heading("V1 PREVIOUSLY OBSERVED PROBLEM CLASSES")
    lines.append("  The following classes caused false positives in the V1 prototype.")
    lines.append("  These are SUPPRESSED in the V2 filter (ignored list in config.yaml).")
    lines.append("  Phase 4 Category 2 tests whether they still appear in raw output.")
    lines.append("")
    lines.append("  Suppressed classes: toothbrush (79), tie (27), remote (65),")
    lines.append("    baseball bat (34), baseball glove (35)")
    lines.append("")
    lines.append("  If these appear in raw detections but NOT in accepted output,")
    lines.append("  the V2 filter is working as intended.")
    lines.append("  If they appear in ACCEPTED output, the filter has a gap — investigate.")
    lines.append("")
    lines.append("  Category 2 observed results:")
    for sc in CATEGORY_2_SCENARIOS:
        sid = sc["id"]
        s = summ_by_id.get(sid)
        lines.append(f"\n    {sc['display_name']}:")
        lines.append(f"      V1 concern: {sc['v1_concern']}")
        if sid in skipped_scenarios:
            lines.append("      SKIPPED")
        elif s is None:
            lines.append("      NOT YET MEASURED")
        else:
            fp = s["fp_observations"]
            supp = s["ignored_suppressed_events"]
            lines.append(f"      FP observations (reached accepted output): {fp}")
            lines.append(f"      Correctly suppressed events: {supp}")
            if s["accepted_class_counts"]:
                lines.append(f"      Accepted classes: {s['accepted_class_counts']}")
            else:
                lines.append(f"      Accepted classes: NONE")

    # ── Limitations ──────────────────────────────────────────────────────────
    heading("KNOWN LIMITATIONS OF THIS EVALUATION")
    lines.append("  1. No labelled image dataset — formal mAP/precision/recall not computed.")
    lines.append("  2. Results reflect the specific room, lighting, and camera position")
    lines.append("     used during testing. Performance in other environments may differ.")
    lines.append("  3. Confidence scores reflect model certainty, not ground-truth correctness.")
    lines.append("  4. FP/minute values describe the tested scene only.")
    lines.append("  5. No tracking used — each frame is evaluated independently.")
    lines.append("  6. 'No-detection observation' on blank frame DOES NOT measure FN rate.")
    lines.append("     See fn_test.py documentation for the distinction.")
    lines.append("  7. Doors and stairs cannot be detected by YOLO11n (COCO gap).")
    lines.append("     This is a documented limitation, not a bug to be worked around.")
    sep()

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# CSV handling
# ─────────────────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "scenario", "category", "timestamp", "ground_truth",
    "detected_class", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "accepted", "eval_result", "rejection_reason", "raw_latency_ms",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4 Detection Evaluation — Interactive Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python tests/evaluation/phase4_eval.py
          python tests/evaluation/phase4_eval.py --frames 30
          python tests/evaluation/phase4_eval.py --fps-duration 60
          python tests/evaluation/phase4_eval.py --categories 1,2
          python tests/evaluation/phase4_eval.py --no-display
          python tests/evaluation/phase4_eval.py --list-scenarios

        Category numbers:
          1 = True Positive Validation
          2 = False Positive Probe (V1 problem objects)
          3 = Empty Scene FP Rate
          4 = Partial Visibility / Occlusion
          5 = Multiple Objects
          6 = Doors and Stairs Gap Analysis
        """),
    )
    parser.add_argument(
        "--frames", type=int, default=90,
        help="Frames per scenario (default: 90, approx 10 seconds)"
    )
    parser.add_argument(
        "--fps-duration", type=int, default=60,
        help="Duration in seconds for Category 3 empty-scene test (default: 60)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV live window (headless mode)"
    )
    parser.add_argument(
        "--categories", type=str, default="",
        help="Comma-separated list of category numbers to run, e.g. '1,2,3'. "
             "Default: run all categories."
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="Print all scenario IDs and exit"
    )
    parser.add_argument(
        "--skip-scenarios", type=str, default="",
        help=(
            "Comma-separated list of scenario IDs to skip entirely, e.g. "
            "'couch_visible,person_standing'. Scenarios marked available=False "
            "in the scenario list are always skipped automatically regardless "
            "of this flag."
        ),
    )
    return parser.parse_args()


def filter_scenarios_by_category(categories_str: str) -> List[dict]:
    """Return only the scenarios belonging to the specified category numbers."""
    if not categories_str.strip():
        return ALL_SCENARIOS
    try:
        nums = {int(c.strip()) for c in categories_str.split(",")}
    except ValueError:
        logger.error("Invalid --categories value: %s", categories_str)
        sys.exit(1)
    return [sc for sc in ALL_SCENARIOS
            if int(sc["category"].split("_")[0]) in nums]


def print_scenario_banner(scenario: dict, index: int, total: int) -> None:
    """Print a clear, beginner-friendly prompt before each scenario."""
    W = 60
    print()
    print("=" * W)
    print(f"  SCENARIO {index}/{total}: {scenario['display_name']}")
    print(f"  Category : {scenario['category']}")
    print(f"  Expected : {scenario.get('expected_class', '?')}")
    if scenario.get("v1_concern"):
        print(f"  V1 concern: {scenario['v1_concern']}")
    if scenario.get("coco_note"):
        print(f"  COCO note : {scenario['coco_note']}")
    print("-" * W)
    print("  Instructions:")
    for line in scenario["instruction"].splitlines():
        print(f"    {line}")
    print("-" * W)
    print("  Press ENTER when ready to start.")
    print("  Press S + ENTER to skip this scenario.")
    print("  Press Q + ENTER to quit the evaluation.")
    print("=" * W)


def run_scenario(
    scenario: dict,
    cam: CameraCapture,
    detector: ObjectDetector,
    det_filter: DetectionFilter,
    n_frames: int,
    fps_duration: Optional[int],
    show_display: bool,
) -> Tuple[List[dict], dict, bool]:
    """
    Run one scenario. Returns (rows, summary, user_quit).

    For Category 3 scenarios, fps_duration overrides n_frames.
    For all others, n_frames is used.

    Returns:
        rows:      List of row dicts for CSV.
        summary:   Summary dict from compute_scenario_summary().
        user_quit: True if the user pressed Q to quit the whole evaluation.
    """
    is_empty_scene = scenario["category"].startswith("3_")

    # Determine loop termination condition
    if is_empty_scene and fps_duration is not None:
        time_limit = float(fps_duration)
        frame_limit = None
    else:
        time_limit = None
        frame_limit = n_frames

    all_rows: List[dict] = []
    frames_processed = 0
    t_start = time.perf_counter()
    user_quit = False

    win_name = "Phase 4 Evaluation — Q to skip"

    while True:
        # Check termination
        elapsed = time.perf_counter() - t_start
        if time_limit is not None and elapsed >= time_limit:
            break
        if frame_limit is not None and frames_processed >= frame_limit:
            break

        frame = cam.read()
        if frame is None:
            logger.warning("Null frame in scenario %s, skipping.", scenario["id"])
            continue

        det_result = detector.detect(frame)
        filtered   = det_filter.filter(det_result)

        frame_rows = evaluate_frame(scenario, filtered, det_result)
        all_rows.extend(frame_rows)
        frames_processed += 1

        if show_display:
            total_shown = frame_limit if frame_limit else int(time_limit or 999)
            display = draw_eval_overlay(
                frame, scenario, filtered, det_result,
                frames_processed, total_shown
            )
            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                user_quit = True
                break

    duration_s = time.perf_counter() - t_start

    if show_display:
        cv2.destroyAllWindows()

    summary = compute_scenario_summary(scenario, all_rows, frames_processed, duration_s)
    return all_rows, summary, user_quit


def main():
    args = parse_args()

    if args.list_scenarios:
        print("\nAll evaluation scenarios:")
        for sc in ALL_SCENARIOS:
            available = sc.get("available", True)
            status = "AVAILABLE" if available else "NOT TESTED"
            print(f"  [{sc['category']}]  {sc['id']}  [{status}]")
            print(f"    Expected: {sc.get('expected_class', '?')}")
            if not available:
                print(f"    Reason: {sc.get('not_tested_reason', '')}")
        return

    config = load_config()

    # Resolve output paths
    eval_dir = _PROJECT_ROOT / "data" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = eval_dir / "phase4_results.csv"
    report_path = eval_dir / "phase4_report.txt"

    scenarios_to_run = filter_scenarios_by_category(args.categories)

    # Build the set of scenario IDs to skip from --skip-scenarios flag
    cli_skipped_ids: set = set()
    if args.skip_scenarios.strip():
        cli_skipped_ids = {
            s.strip() for s in args.skip_scenarios.split(",") if s.strip()
        }

    # Scenarios marked available=False are ALWAYS skipped, regardless of CLI.
    # They are recorded separately as NOT_TESTED (different from user-skipped).
    not_tested = [
        sc for sc in scenarios_to_run if not sc.get("available", True)
    ]
    not_tested_ids = {sc["id"] for sc in not_tested}

    # Active scenarios: available AND not in the CLI skip list
    active_scenarios = [
        sc for sc in scenarios_to_run
        if sc.get("available", True) and sc["id"] not in cli_skipped_ids
    ]

    print()
    print("=" * 60)
    print("  PHASE 4 — DETECTION EVALUATION")
    print("=" * 60)
    print(f"  Model            : {config['detection']['model_name']}")
    print(f"  Conf threshold   : {config['detection']['confidence_threshold']}")
    print(f"  Frames/scenario  : {args.frames}")
    print(f"  Empty scene time : {args.fps_duration}s")
    print(f"  Active scenarios : {len(active_scenarios)}")
    if not_tested_ids:
        print(f"  NOT TESTED (unavailable): {', '.join(not_tested_ids)}")
    if cli_skipped_ids:
        print(f"  Skipped (--skip-scenarios): {', '.join(cli_skipped_ids)}")
    print(f"  CSV output       : {csv_path}")
    print(f"  Report output    : {report_path}")
    print()
    print("  IMPORTANT: This evaluation requires your physical presence.")
    print("  You will need to place objects in front of the camera for each")
    print("  scenario. Follow the instructions shown before each test.")
    print()
    print("  Nothing is pre-populated. Results come from actual camera data.")
    print()
    input("  Press ENTER to load the model and open the camera...")

    # ── Load model ───────────────────────────────────────────────────────────
    print("\nLoading YOLO model (may take a few seconds) ...")
    try:
        detector = ObjectDetector(config)
        detector.load()
        print(f"Model loaded: {detector.model_name}")
    except DetectorError as e:
        print(f"ERROR: Could not load model: {e}")
        sys.exit(1)

    det_filter = DetectionFilter(config)

    # ── Open camera ──────────────────────────────────────────────────────────
    print("Opening camera ...")
    try:
        cam = CameraCapture(config)
        cam.open()
        print(f"Camera ready: {cam.actual_width}x{cam.actual_height}")
    except CameraError as e:
        print(f"ERROR: Camera problem: {e}")
        sys.exit(1)

    # ── CSV setup ────────────────────────────────────────────────────────────
    csv_file   = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
    csv_writer.writeheader()

    # ── Run scenarios ────────────────────────────────────────────────────────
    all_summaries  = []
    skipped        = []   # user pressed S during the run
    user_quit_eval = False

    try:
        for i, scenario in enumerate(active_scenarios, start=1):
            print_scenario_banner(scenario, i, len(active_scenarios))

            user_input = input("  > ").strip().lower()
            if user_input == "q":
                print("  Quitting evaluation.")
                break
            if user_input == "s":
                print(f"  Skipping: {scenario['id']}")
                skipped.append(scenario["id"])
                continue

            # Run the scenario
            rows, summary, user_quit = run_scenario(
                scenario=scenario,
                cam=cam,
                detector=detector,
                det_filter=det_filter,
                n_frames=args.frames,
                fps_duration=args.fps_duration,
                show_display=not args.no_display,
            )

            # Write rows to CSV
            for row in rows:
                csv_writer.writerow(row)
            csv_file.flush()

            all_summaries.append(summary)

            # Print scenario summary
            print(f"\n  --- Scenario complete: {scenario['id']} ---")
            print(f"      Frames processed       : {summary['n_frames']}")
            print(f"      Duration               : {summary['duration_s']:.1f}s")
            print(f"      Correct-class obs.     : {summary['correct_class_observations']}")
            print(f"      Wrong-class obs.       : {summary['wrong_class_observations']}")
            print(f"      No-detection obs.      : {summary['no_detection_observations']}")
            print(f"      FP observations        : {summary['fp_observations']}")
            print(f"      Suppressed (ignored)   : {summary['ignored_suppressed_events']}")
            if summary["accepted_class_counts"]:
                print(f"      Accepted classes       : {summary['accepted_class_counts']}")
            if "conf_mean" in summary:
                print(f"      Confidence mean        : {summary['conf_mean']:.3f}")
            print()

            if user_quit:
                print("  User pressed Q — ending evaluation early.")
                break

    except KeyboardInterrupt:
        print("\n  Interrupted by Ctrl+C — saving results.")

    finally:
        cam.release()
        csv_file.close()

    # ── Write report ─────────────────────────────────────────────────────────
    print(f"\nWriting evaluation report to: {report_path}")
    build_report(
        all_summaries, config, report_path,
        skipped_scenarios=skipped,
        not_tested_ids=not_tested_ids,
        cli_skipped_ids=cli_skipped_ids,
    )

    # ── Final console summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Scenarios measured     : {len(all_summaries)}")
    print(f"  Skipped (user, S key)  : {len(skipped)}")
    print(f"  Skipped (--skip-scenarios): {len(cli_skipped_ids)}")
    print(f"  NOT TESTED (unavailable)  : {len(not_tested_ids)}")
    print(f"  CSV                    : {csv_path}")
    print(f"  Report                 : {report_path}")
    print()
    remaining = set(sc["id"] for sc in active_scenarios) - {s["scenario_id"] for s in all_summaries} - set(skipped)
    if remaining:
        print("  Scenarios not yet run:")
        for sid in sorted(remaining):
            print(f"    {sid}")
        print()
        print("  To run remaining scenarios:")
        print("    python tests/evaluation/phase4_eval.py")
    print("  To view the report:")
    print(f"    type {report_path}")
    print()


if __name__ == "__main__":
    main()
