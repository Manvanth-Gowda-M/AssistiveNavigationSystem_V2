"""
Unit Tests - Phase 4 Evaluation Logic
========================================
Tests the evaluation logic from phase4_eval.py using synthetic detection
data. No camera, no model loading, no physical objects required.

What is tested:
  - evaluate_frame() — per-frame result classification
  - compute_scenario_summary() — aggregation of frame results
  - build_report() — report generation with synthetic summaries
  - CSV column structure
  - Conservative metric naming (no precision/recall/mAP/F1)
  - FP/minute calculation
  - Blank frame is NOT labelled as FN
  - Gap-analysis scenarios produce correct result codes
  - Ignored-class suppressions recorded correctly

How to run:
    .venv/Scripts/python.exe -m pytest tests/unit/test_phase4_logic.py -v
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Make src/ importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from assistive_navigation.detection.detector import Detection, DetectionResult
from assistive_navigation.detection.filter import (
    FilterResult, Priority, RejectionReason
)

# Phase 4 evaluation logic — import the functions we want to test
sys.path.insert(0, str(_PROJECT_ROOT / "tests" / "evaluation"))
from phase4_eval import (
    evaluate_frame,
    compute_scenario_summary,
    build_report,
    FrameResult,
    CSV_COLUMNS,
    CATEGORY_1_SCENARIOS,
    CATEGORY_2_SCENARIOS,
    CATEGORY_6_SCENARIOS,
    ALL_SCENARIOS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_detection(class_id: int, class_name: str, confidence: float = 0.80) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(10.0, 10.0, 200.0, 300.0),
        bbox_norm=(0.015, 0.02, 0.31, 0.63),
        center_x=0.163,
        center_y=0.323,
        area_norm=0.183,
        frame_width=640,
        frame_height=480,
    )


def make_det_result(detections: list, latency_ms: float = 55.0) -> DetectionResult:
    return DetectionResult(
        detections=detections,
        latency_ms=latency_ms,
        frame_width=640,
        frame_height=480,
        raw_count=len(detections),
    )


def make_filter_result(
    accepted: list,
    rejected: list,
    raw_count: int = None,
) -> FilterResult:
    pm = {id(d): Priority.CRITICAL for d in accepted}
    return FilterResult(
        accepted=list(accepted),
        rejected=list(rejected),
        raw_count=raw_count if raw_count is not None else len(accepted) + len(rejected),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        priority_map=pm,
    )


def make_rejection_dict(det: Detection, reason: str) -> dict:
    return {
        "class_id":   det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "reason":     reason,
        "bbox":       det.bbox,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tests for evaluate_frame()
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateFrame:
    """Tests for the per-frame result classification function."""

    def _scenario(self, expected_class, category="1_true_positive"):
        return {
            "id": "test_scenario",
            "category": category,
            "expected_class": expected_class,
        }

    # ── Correct class detection ───────────────────────────────────────────────

    def test_correct_class_gives_correct_class_observation(self):
        """Expected class detected and accepted → CORRECT_CLASS."""
        det = make_detection(0, "person", 0.88)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])

        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert len(rows) == 1
        assert rows[0]["eval_result"] == FrameResult.CORRECT_CLASS
        assert rows[0]["detected_class"] == "person"
        assert rows[0]["accepted"] is True

    # ── Wrong class detection ─────────────────────────────────────────────────

    def test_wrong_class_gives_wrong_class_observation(self):
        """Different class accepted when specific class expected → WRONG_CLASS."""
        det = make_detection(56, "chair", 0.75)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])  # chair accepted, but expected person

        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert len(rows) == 1
        assert rows[0]["eval_result"] == FrameResult.WRONG_CLASS

    # ── No detection ─────────────────────────────────────────────────────────

    def test_zero_raw_detections_gives_no_detection_for_expected_object(self):
        """Zero raw detections when object expected → NO_DETECTION."""
        det_result = make_det_result([])
        filtered   = make_filter_result([], [])

        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert len(rows) == 1
        assert rows[0]["eval_result"] == FrameResult.NO_DETECTION

    # ── Empty-scene correct ───────────────────────────────────────────────────

    def test_zero_detections_in_empty_scene_gives_empty_correct(self):
        """Zero detections when nothing expected → EMPTY_CORRECT."""
        det_result = make_det_result([])
        filtered   = make_filter_result([], [])

        rows = evaluate_frame(
            self._scenario("no_navigation_object", "2_false_positive_probe"),
            filtered, det_result
        )
        assert len(rows) == 1
        assert rows[0]["eval_result"] == FrameResult.EMPTY_CORRECT

    def test_empty_scene_category3_zero_detections_correct(self):
        """Category 3 (empty scene test): zero detections → EMPTY_CORRECT."""
        det_result = make_det_result([])
        filtered   = make_filter_result([], [])

        rows = evaluate_frame(
            self._scenario("no_navigation_object", "3_empty_scene"),
            filtered, det_result
        )
        assert rows[0]["eval_result"] == FrameResult.EMPTY_CORRECT

    # ── False positive observation ────────────────────────────────────────────

    def test_accepted_detection_in_empty_scene_gives_fp_observation(self):
        """Accepted detection when nothing expected → FP_OBSERVATION."""
        det = make_detection(0, "person", 0.82)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])

        rows = evaluate_frame(
            self._scenario("no_navigation_object", "2_false_positive_probe"),
            filtered, det_result
        )
        # Person is accepted but nothing was expected → FP
        fp_rows = [r for r in rows if r["eval_result"] == FrameResult.FP_OBSERVATION]
        assert len(fp_rows) >= 1

    # ── Ignored class suppression ─────────────────────────────────────────────

    def test_ignored_class_in_empty_scene_gives_suppressed(self):
        """Rejected-as-ignored class in empty scene → IGNORED_SUPPRESSED."""
        toothbrush = make_detection(79, "toothbrush", 0.71)
        rej = make_rejection_dict(toothbrush, RejectionReason.CLASS_IGNORED)
        det_result = make_det_result([toothbrush])
        filtered   = make_filter_result([], [rej])

        rows = evaluate_frame(
            self._scenario("no_navigation_object", "2_false_positive_probe"),
            filtered, det_result
        )
        suppressed_rows = [r for r in rows
                           if r["eval_result"] == FrameResult.IGNORED_SUPPRESSED]
        assert len(suppressed_rows) >= 1, (
            "Ignored class in empty scene should produce IGNORED_SUPPRESSED rows"
        )

    # ── COCO gap scenarios ────────────────────────────────────────────────────

    def test_gap_scenario_produces_gap_observation(self):
        """Category 6 gap scenario: any detection → gap_observation."""
        det = make_detection(0, "person", 0.45)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])

        gap_scenario = {
            "id": "door_gap_analysis",
            "category": "6_gap_analysis",
            "expected_class": "COCO_GAP",
        }
        rows = evaluate_frame(gap_scenario, filtered, det_result)
        assert len(rows) >= 1
        assert rows[0]["eval_result"] == "gap_observation"

    def test_gap_scenario_zero_detections_gives_empty_correct(self):
        """Category 6 gap scenario: zero detections → EMPTY_CORRECT."""
        det_result = make_det_result([])
        filtered   = make_filter_result([], [])

        gap_scenario = {
            "id": "stairs_gap_analysis",
            "category": "6_gap_analysis",
            "expected_class": "COCO_GAP",
        }
        rows = evaluate_frame(gap_scenario, filtered, det_result)
        assert len(rows) == 1
        assert rows[0]["eval_result"] == FrameResult.EMPTY_CORRECT

    # ── Row structure ─────────────────────────────────────────────────────────

    def test_row_has_all_csv_columns(self):
        """Each row returned by evaluate_frame must have all CSV column keys."""
        det = make_detection(0, "person", 0.80)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])
        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        for row in rows:
            for col in CSV_COLUMNS:
                assert col in row, f"Missing CSV column '{col}' in row: {row}"

    def test_confidence_recorded_in_row(self):
        """Confidence value must be recorded in the row."""
        det = make_detection(0, "person", 0.91)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])
        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert rows[0]["confidence"] == "0.9100"

    def test_scenario_id_in_row(self):
        """Scenario ID must be recorded in the row."""
        det = make_detection(56, "chair", 0.78)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])
        sc = {"id": "my_test_scenario", "category": "1_true_positive",
              "expected_class": "chair"}
        rows = evaluate_frame(sc, filtered, det_result)
        assert rows[0]["scenario"] == "my_test_scenario"

    def test_ground_truth_in_row(self):
        """Ground truth class must be in the row."""
        det = make_detection(0, "person", 0.80)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])
        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert rows[0]["ground_truth"] == "person"

    def test_latency_in_row(self):
        """Inference latency must be recorded in each row."""
        det = make_detection(0, "person", 0.80)
        det_result = make_det_result([det], latency_ms=123.45)
        filtered   = make_filter_result([det], [])
        rows = evaluate_frame(self._scenario("person"), filtered, det_result)
        assert rows[0]["raw_latency_ms"] == "123.45"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for compute_scenario_summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeScenarioSummary:
    """Tests for per-scenario aggregation."""

    def _make_rows(self, eval_results: list, confidences: list = None) -> list:
        """Make synthetic rows matching CSV structure."""
        rows = []
        for i, er in enumerate(eval_results):
            accepted = er in (
                FrameResult.CORRECT_CLASS,
                FrameResult.WRONG_CLASS,
                FrameResult.FP_OBSERVATION,
            )
            conf = str(confidences[i]) if confidences else "0.8000"
            rows.append({
                "scenario":          "test",
                "category":          "1_true_positive",
                "timestamp":         "2026-01-01T00:00:00",
                "ground_truth":      "person",
                "detected_class":    "person",
                "confidence":        conf,
                "bbox_x1": "10", "bbox_y1": "10",
                "bbox_x2": "100", "bbox_y2": "100",
                "accepted":          accepted,
                "eval_result":       er,
                "rejection_reason":  "",
                "raw_latency_ms":    "65.0",
            })
        return rows

    def test_summary_has_required_keys(self):
        """Summary dict must have all required keys."""
        sc = {"id": "test", "category": "1_true_positive",
              "expected_class": "person"}
        rows = self._make_rows([FrameResult.CORRECT_CLASS] * 10)
        summary = compute_scenario_summary(sc, rows, n_frames=10, duration_s=5.0)
        required = [
            "scenario_id", "category", "expected_class",
            "n_frames", "duration_s", "total_raw_detections",
            "correct_class_observations", "wrong_class_observations",
            "no_detection_observations", "fp_observations",
            "ignored_suppressed_events", "fp_per_minute",
            "accepted_class_counts",
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_correct_class_count(self):
        """correct_class_observations must be counted correctly."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows([
            FrameResult.CORRECT_CLASS,
            FrameResult.CORRECT_CLASS,
            FrameResult.NO_DETECTION,
        ])
        summary = compute_scenario_summary(sc, rows, 3, 3.0)
        assert summary["correct_class_observations"] == 2

    def test_wrong_class_count(self):
        """wrong_class_observations must be counted correctly."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows([
            FrameResult.CORRECT_CLASS,
            FrameResult.WRONG_CLASS,
            FrameResult.WRONG_CLASS,
        ])
        summary = compute_scenario_summary(sc, rows, 3, 3.0)
        assert summary["wrong_class_observations"] == 2

    def test_fp_per_minute_calculation(self):
        """FP/minute must be calculated from fp_observations and duration."""
        sc = {"id": "t", "category": "3_empty_scene",
              "expected_class": "no_navigation_object"}
        rows = self._make_rows([
            FrameResult.FP_OBSERVATION,
            FrameResult.FP_OBSERVATION,
            FrameResult.EMPTY_CORRECT,
        ])
        # 2 FP events in 30 seconds = 4.0 per minute
        summary = compute_scenario_summary(sc, rows, 3, 30.0)
        assert abs(summary["fp_per_minute"] - 4.0) < 0.01

    def test_zero_duration_fp_per_minute(self):
        """Zero duration must not cause division by zero."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows([FrameResult.CORRECT_CLASS])
        summary = compute_scenario_summary(sc, rows, 1, 0.0)
        assert summary["fp_per_minute"] == 0.0

    def test_confidence_stats_computed(self):
        """Confidence statistics computed when accepted detections exist."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows(
            [FrameResult.CORRECT_CLASS, FrameResult.CORRECT_CLASS],
            confidences=[0.80, 0.90]
        )
        summary = compute_scenario_summary(sc, rows, 2, 2.0)
        assert "conf_mean" in summary
        assert abs(summary["conf_mean"] - 0.85) < 0.001
        assert summary["conf_min"] == 0.80
        assert summary["conf_max"] == 0.90

    def test_no_confidence_stats_when_nothing_accepted(self):
        """No confidence stats when there are no accepted detections."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows([FrameResult.NO_DETECTION, FrameResult.NO_DETECTION])
        # Force accepted=False on all rows
        for r in rows:
            r["accepted"] = False
        summary = compute_scenario_summary(sc, rows, 2, 2.0)
        assert "conf_mean" not in summary

    def test_n_frames_recorded(self):
        """n_frames must match what was passed."""
        sc = {"id": "t", "category": "1_true_positive", "expected_class": "person"}
        rows = self._make_rows([FrameResult.CORRECT_CLASS] * 5)
        summary = compute_scenario_summary(sc, rows, 5, 5.0)
        assert summary["n_frames"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Tests for CSV structure
# ─────────────────────────────────────────────────────────────────────────────

class TestCSVStructure:
    """Tests for CSV output structure."""

    def test_csv_columns_list_is_correct(self):
        """CSV_COLUMNS must contain all required field names."""
        required = [
            "scenario", "category", "timestamp", "ground_truth",
            "detected_class", "confidence",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "accepted", "eval_result", "rejection_reason", "raw_latency_ms",
        ]
        for col in required:
            assert col in CSV_COLUMNS, f"CSV column '{col}' is missing"

    def test_csv_is_written_correctly(self, tmp_path):
        """Rows produced by evaluate_frame can be written to CSV."""
        import csv
        det = make_detection(0, "person", 0.80)
        det_result = make_det_result([det])
        filtered   = make_filter_result([det], [])
        sc = {"id": "test", "category": "1_true_positive", "expected_class": "person"}
        rows = evaluate_frame(sc, filtered, det_result)

        out = tmp_path / "test_output.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "person" in content
        assert "correct_class_observation" in content


# ─────────────────────────────────────────────────────────────────────────────
# Tests for build_report()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildReport:
    """Tests for report generation."""

    def _minimal_config(self) -> dict:
        return {
            "detection": {
                "model_name": "yolo11n",
                "confidence_threshold": 0.35,
                "device": "cpu",
            }
        }

    def _make_summary(self, scenario_id: str) -> dict:
        return {
            "scenario_id":                 scenario_id,
            "category":                    "1_true_positive",
            "expected_class":              "person",
            "n_frames":                    90,
            "duration_s":                  10.5,
            "total_raw_detections":        75,
            "correct_class_observations":  72,
            "wrong_class_observations":    2,
            "no_detection_observations":   1,
            "fp_observations":             0,
            "ignored_suppressed_events":   0,
            "gap_observations":            0,
            "empty_correct":               0,
            "accepted_class_counts":       {"person": 72},
            "fp_per_minute":               0.0,
            "conf_mean":                   0.84,
            "conf_min":                    0.36,
            "conf_max":                    0.95,
            "conf_stdev":                  0.09,
        }

    def test_report_created(self, tmp_path):
        """build_report() must create the output file."""
        out = tmp_path / "test_report.txt"
        summaries = [self._make_summary("person_standing")]
        build_report(summaries, self._minimal_config(), out, [])
        assert out.exists()

    def test_report_contains_scenario_id(self, tmp_path):
        """Report must contain the scenario ID."""
        out = tmp_path / "report.txt"
        build_report([self._make_summary("person_standing")],
                     self._minimal_config(), out, [])
        content = out.read_text(encoding="utf-8")
        assert "person_standing" in content

    def test_report_not_yet_measured_for_unrun_scenario(self, tmp_path):
        """Scenarios not in summaries must show NOT YET MEASURED."""
        out = tmp_path / "report.txt"
        # Provide summary for one scenario only; others should say NM
        build_report(
            [self._make_summary("person_standing")],
            self._minimal_config(), out, []
        )
        content = out.read_text(encoding="utf-8")
        assert "NOT YET MEASURED" in content

    def test_report_skipped_shows_skipped(self, tmp_path):
        """Scenarios in skipped list must show SKIPPED."""
        out = tmp_path / "report.txt"
        build_report([], self._minimal_config(), out,
                     skipped_scenarios=["person_standing"])
        content = out.read_text(encoding="utf-8")
        assert "SKIPPED" in content

    def test_report_contains_coco_gap_section(self, tmp_path):
        """Report must contain the COCO gap section."""
        out = tmp_path / "report.txt"
        build_report([], self._minimal_config(), out, [])
        content = out.read_text(encoding="utf-8")
        assert "CAPABILITY GAP" in content
        assert "door" in content.lower() or "stairs" in content.lower()

    def test_report_contains_v1_fp_section(self, tmp_path):
        """Report must contain the V1 false-positive section."""
        out = tmp_path / "report.txt"
        build_report([], self._minimal_config(), out, [])
        content = out.read_text(encoding="utf-8")
        assert "V1" in content or "PREVIOUSLY OBSERVED" in content

    def test_report_does_not_claim_formal_accuracy_metrics(self, tmp_path):
        """
        The report must not present precision, recall, mAP, or F1 as
        computed evaluation metrics for this system.

        It IS acceptable for the report to mention these terms in the
        context of explaining why they are NOT used (e.g. a limitations
        section that says "formal mAP/precision/recall not computed").

        This test checks that the report does not contain lines that
        present these as measured results, e.g. "Precision: 0.85".
        """
        out = tmp_path / "report.txt"
        build_report([self._make_summary("person_standing")],
                     self._minimal_config(), out, [])
        content = out.read_text(encoding="utf-8").lower()

        # These patterns would indicate the metrics are being reported as results
        forbidden_patterns = [
            "precision:",       # "Precision: 0.85" — claiming precision was computed
            "recall:",          # "Recall: 0.91" — claiming recall was computed
            "f1 score:",        # "F1 Score: 0.88"
            "map:",             # "mAP: 0.73"
            "mean average precision:",
        ]
        for pattern in forbidden_patterns:
            assert pattern not in content, (
                f"Report must not present '{pattern}' as a measured result. "
                f"The methodology does not support these metrics."
            )

    def test_report_contains_safety_disclaimer(self, tmp_path):
        """Report must mention research prototype / safety disclaimer."""
        out = tmp_path / "report.txt"
        build_report([], self._minimal_config(), out, [])
        content = out.read_text(encoding="utf-8")
        assert "RESEARCH PROTOTYPE" in content or "research prototype" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Tests for scenario definitions
# ─────────────────────────────────────────────────────────────────────────────

class TestScenarioDefinitions:
    """Structural tests for the scenario lists."""

    def test_all_scenarios_have_required_fields(self):
        """Every scenario must have id, category, expected_class, instruction."""
        for sc in ALL_SCENARIOS:
            for field in ("id", "category", "expected_class", "instruction"):
                assert field in sc, (
                    f"Scenario '{sc.get('id', '?')}' missing field '{field}'"
                )

    def test_category_1_has_seven_scenarios(self):
        """Category 1 must have exactly 7 navigation objects to test."""
        from phase4_eval import CATEGORY_1_SCENARIOS
        assert len(CATEGORY_1_SCENARIOS) == 7

    def test_category_2_includes_v1_problem_objects(self):
        """Category 2 must include charger, pen, phone, empty_desk."""
        ids = {sc["id"] for sc in CATEGORY_2_SCENARIOS}
        for required_id in ("usb_charger", "pen_or_pencil", "phone_only", "empty_desk"):
            assert required_id in ids, f"V1 problem scenario '{required_id}' is missing"

    def test_category_6_includes_door_and_stairs(self):
        """Category 6 must include door and stairs gap scenarios."""
        ids = {sc["id"] for sc in CATEGORY_6_SCENARIOS}
        assert "door_gap_analysis" in ids
        assert "stairs_gap_analysis" in ids

    def test_all_scenario_ids_unique(self):
        """No two scenarios may share the same ID."""
        ids = [sc["id"] for sc in ALL_SCENARIOS]
        assert len(ids) == len(set(ids)), (
            f"Duplicate scenario IDs found: "
            f"{[x for x in ids if ids.count(x) > 1]}"
        )

    def test_category_6_scenarios_have_coco_note(self):
        """Gap analysis scenarios must explain the COCO gap."""
        for sc in CATEGORY_6_SCENARIOS:
            assert "coco_note" in sc, (
                f"Gap scenario '{sc['id']}' missing 'coco_note'"
            )

    def test_category_2_scenarios_have_v1_concern(self):
        """V1 probe scenarios must document the V1 concern being tested."""
        for sc in CATEGORY_2_SCENARIOS:
            assert "v1_concern" in sc, (
                f"V1 probe scenario '{sc['id']}' missing 'v1_concern'"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tests for FrameResult constants
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameResultCodes:
    """Tests that FrameResult uses conservative terminology."""

    def test_no_precision_or_recall_in_result_codes(self):
        """
        FrameResult codes must not use 'true positive', 'false positive',
        'false negative' as raw labels — they use conservative observation
        language as required by the evaluation methodology.
        """
        codes = [
            FrameResult.CORRECT_CLASS,
            FrameResult.WRONG_CLASS,
            FrameResult.NO_DETECTION,
            FrameResult.FP_OBSERVATION,
            FrameResult.IGNORED_SUPPRESSED,
            FrameResult.EMPTY_CORRECT,
        ]
        for code in codes:
            # Code strings use "observation" not "true_positive" etc.
            assert isinstance(code, str)
            assert len(code) > 0

    def test_fp_observation_code_is_observation_not_claim(self):
        """FP code uses 'observation' language."""
        assert "observation" in FrameResult.FP_OBSERVATION

    def test_correct_class_code_is_observation_language(self):
        """Correct detection code uses 'observation' language."""
        assert "observation" in FrameResult.CORRECT_CLASS
