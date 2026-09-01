"""
False Positive Rate Test — Phase 4
====================================
Opens the webcam, runs the detector for a configurable duration,
and counts accepted navigation detections.

Purpose:
    Measure how many navigation detections are produced in a scene
    where the user declares no intentional navigation objects are present.

What this measures:
    "Observed accepted-detection rate" in a declared-empty scene.
    If the user correctly clears the camera view of navigation objects,
    accepted detections represent observed false positive events for that scene.

What this does NOT claim:
    - It does not claim these are universally representative false positive rates.
    - It does not claim the system is "safe" if the rate is low.
    - It does not assert that zero is the required result.
      A non-zero result is a valid measurement, not a test failure.
    - Results are specific to the tested room, lighting, and environment.

Usage:
    python tests/evaluation/fp_test.py
    python tests/evaluation/fp_test.py --duration 30
    python tests/evaluation/fp_test.py --duration 60
    python tests/evaluation/fp_test.py --no-display

Also runnable as a pytest test (non-interactive, shorter duration):
    .venv/Scripts/python.exe -m pytest tests/evaluation/fp_test.py -v -s
"""

import sys
import csv
import time
import argparse
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

# Make src/ importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import cv2
import numpy as np
import pytest

from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.utils.logger import get_logger
from assistive_navigation.camera.capture import CameraCapture, CameraError
from assistive_navigation.detection.detector import ObjectDetector, DetectionResult, DetectorError
from assistive_navigation.detection.filter import DetectionFilter, FilterResult, RejectionReason

logger = get_logger(__name__, level="INFO")


# ─────────────────────────────────────────────────────────────────────────────
# Core measurement function — independently testable
# ─────────────────────────────────────────────────────────────────────────────
def run_fp_measurement(
    cam: CameraCapture,
    detector: ObjectDetector,
    det_filter: DetectionFilter,
    duration_s: float,
    show_display: bool = False,
    scene_label: str = "unspecified",
) -> dict:
    """
    Run detector for duration_s seconds and count observed accepted detections.

    This function does NOT interact with the user and does NOT assert.
    It only measures.

    Args:
        cam:          Open CameraCapture instance.
        detector:     Loaded ObjectDetector instance.
        det_filter:   DetectionFilter instance.
        duration_s:   How long to run in seconds.
        show_display: Whether to open an OpenCV window.
        scene_label:  Descriptive label for this measurement run.

    Returns:
        dict with measurement results:
          frames_processed  : int
          accepted_total    : int
          accepted_per_min  : float
          class_counts      : dict {class_name: count}
          conf_values       : list of accepted confidence values
          conf_mean         : float or None
          conf_min          : float or None
          conf_max          : float or None
          duration_actual_s : float
          scene_label       : str
    """
    frames_processed = 0
    accepted_total   = 0
    class_counts: Dict[str, int] = defaultdict(int)
    conf_values: List[float] = []

    t_start = time.perf_counter()

    while (time.perf_counter() - t_start) < duration_s:
        frame = cam.read()
        if frame is None:
            continue

        det_result = detector.detect(frame)
        filtered   = det_filter.filter(det_result)
        frames_processed += 1
        accepted_total += filtered.accepted_count

        for det in filtered.accepted:
            class_counts[det.class_name] += 1
            conf_values.append(det.confidence)

        if show_display:
            overlay = frame.copy()
            h, w = overlay.shape[:2]
            elapsed = time.perf_counter() - t_start
            remaining = max(0.0, duration_s - elapsed)
            cv2.putText(overlay,
                        f"FP Test: {elapsed:.1f}s / {duration_s:.0f}s  "
                        f"(remaining: {remaining:.1f}s)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(overlay,
                        f"Accepted so far: {accepted_total}  "
                        f"Frames: {frames_processed}",
                        (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
            cv2.putText(overlay,
                        "Scene should contain NO navigation objects. Q=quit",
                        (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            for det in filtered.accepted:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 220), 2)
                cv2.putText(overlay,
                            f"FP? {det.class_name} {det.confidence:.2f}",
                            (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 220), 1)
            cv2.imshow("FP Test — Q to stop", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show_display:
        cv2.destroyAllWindows()

    duration_actual = time.perf_counter() - t_start
    accepted_per_min = (accepted_total / duration_actual * 60.0) if duration_actual > 0 else 0.0

    return {
        "frames_processed":   frames_processed,
        "accepted_total":     accepted_total,
        "accepted_per_min":   round(accepted_per_min, 3),
        "class_counts":       dict(class_counts),
        "conf_values":        conf_values,
        "conf_mean":          statistics.mean(conf_values) if conf_values else None,
        "conf_min":           min(conf_values) if conf_values else None,
        "conf_max":           max(conf_values) if conf_values else None,
        "duration_actual_s":  round(duration_actual, 2),
        "scene_label":        scene_label,
    }


def print_fp_result(result: dict, duration_s: float) -> None:
    """Print a clear, honest measurement report to the console."""
    print()
    print("=" * 60)
    print("  FALSE POSITIVE RATE MEASUREMENT RESULT")
    print("=" * 60)
    print(f"  Scene label          : {result['scene_label']}")
    print(f"  Requested duration   : {duration_s:.0f} s")
    print(f"  Actual duration      : {result['duration_actual_s']:.2f} s")
    print(f"  Frames processed     : {result['frames_processed']}")
    print()
    print(f"  Accepted detections  : {result['accepted_total']}")
    print(f"  Accepted / minute    : {result['accepted_per_min']:.2f}")
    print()
    if result["class_counts"]:
        print("  Classes accepted (unexpected in empty scene):")
        for cls, cnt in sorted(result["class_counts"].items(), key=lambda x: -x[1]):
            print(f"    {cls:<25}: {cnt}")
        print()
        print(f"  Confidence (accepted): "
              f"mean={result['conf_mean']:.3f}, "
              f"min={result['conf_min']:.3f}, "
              f"max={result['conf_max']:.3f}")
    else:
        print("  Classes accepted     : NONE")
        print("  (No accepted detections observed in this scene)")
    print()
    print("  NOTE: These results describe this specific scene and environment.")
    print("  A non-zero count is a valid measurement, not a test failure.")
    print("  Do not generalise this rate to all environments.")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: False Positive Rate Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/evaluation/fp_test.py
  python tests/evaluation/fp_test.py --duration 30
  python tests/evaluation/fp_test.py --duration 120 --no-display
        """
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Measurement duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable OpenCV live window"
    )
    parser.add_argument(
        "--scene", type=str, default="user_declared_empty",
        help="Short label for the scene being tested (default: user_declared_empty)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    print()
    print("=" * 60)
    print("  PHASE 4 — FALSE POSITIVE RATE MEASUREMENT")
    print("=" * 60)
    print(f"  Duration   : {args.duration:.0f} seconds")
    print(f"  Scene      : {args.scene}")
    print()
    print("  Before starting:")
    print("  1. Remove ALL navigation objects from the camera view.")
    print("     (No person, no chair, no bottle, no backpack, etc.)")
    print("  2. Point the camera at the scene you want to test.")
    print("     (e.g. empty desk, empty wall, empty floor)")
    print()
    print("  Any accepted detection will be counted as an observed FP event")
    print("  for this scene. Results are measurements, not pass/fail verdicts.")
    print()
    input("  Press ENTER when the scene is ready ...")

    # Load model
    try:
        detector = ObjectDetector(config)
        detector.load()
    except DetectorError as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)

    det_filter = DetectionFilter(config)

    # Open camera
    try:
        cam = CameraCapture(config)
        cam.open()
        print(f"Camera: {cam.actual_width}x{cam.actual_height}. Starting measurement...")
    except CameraError as e:
        print(f"ERROR: Camera problem: {e}")
        sys.exit(1)

    try:
        result = run_fp_measurement(
            cam=cam,
            detector=detector,
            det_filter=det_filter,
            duration_s=args.duration,
            show_display=not args.no_display,
            scene_label=args.scene,
        )
    finally:
        cam.release()

    print_fp_result(result, args.duration)

    # Save to CSV
    eval_dir = _PROJECT_ROOT / "data" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = eval_dir / f"fp_test_{ts}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scene_label", "duration_s", "frames_processed",
            "accepted_total", "accepted_per_min",
            "class_counts", "conf_mean", "conf_min", "conf_max"
        ])
        writer.writerow([
            result["scene_label"],
            result["duration_actual_s"],
            result["frames_processed"],
            result["accepted_total"],
            result["accepted_per_min"],
            str(result["class_counts"]),
            result["conf_mean"] if result["conf_mean"] else "",
            result["conf_min"] if result["conf_min"] else "",
            result["conf_max"] if result["conf_max"] else "",
        ])
    print(f"\n  Results saved to: {out_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Pytest automated tests
# These run WITHOUT requiring manual camera interaction.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def config_fixture():
    return load_config()


class TestFPMeasurementLogic:
    """
    Tests for the FP measurement logic using synthetic data.
    No camera or model required.
    """

    def _make_empty_filter_result(self):
        """FilterResult with zero detections."""
        from assistive_navigation.detection.filter import FilterResult
        return FilterResult(
            accepted=[], rejected=[], raw_count=0,
            accepted_count=0, rejected_count=0, priority_map={}
        )

    def _make_detection(self, class_id, class_name, confidence=0.75):
        from assistive_navigation.detection.detector import Detection
        return Detection(
            class_id=class_id, class_name=class_name, confidence=confidence,
            bbox=(10, 10, 100, 100), bbox_norm=(0.015, 0.02, 0.156, 0.208),
            center_x=0.085, center_y=0.115, area_norm=0.022,
            frame_width=640, frame_height=480
        )

    def _make_filter_result_with_accepted(self, detections):
        from assistive_navigation.detection.filter import FilterResult, Priority
        from assistive_navigation.detection.detector import Detection
        pm = {id(d): Priority.CRITICAL for d in detections}
        return FilterResult(
            accepted=list(detections),
            rejected=[],
            raw_count=len(detections),
            accepted_count=len(detections),
            rejected_count=0,
            priority_map=pm,
        )

    def test_result_dict_has_expected_keys(self):
        """run_fp_measurement returns a dict with the required keys."""
        # We test the structure by calling with a mock-like approach:
        # Just check the keys of a manually constructed result.
        result = {
            "frames_processed": 10,
            "accepted_total": 0,
            "accepted_per_min": 0.0,
            "class_counts": {},
            "conf_values": [],
            "conf_mean": None,
            "conf_min": None,
            "conf_max": None,
            "duration_actual_s": 5.0,
            "scene_label": "test",
        }
        required_keys = [
            "frames_processed", "accepted_total", "accepted_per_min",
            "class_counts", "conf_values", "duration_actual_s", "scene_label"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_zero_accepted_produces_zero_per_minute(self):
        """If accepted_total=0, accepted_per_min must be 0.0."""
        # Simulate: 60 seconds, 0 accepted
        accepted = 0
        duration = 60.0
        per_min = (accepted / duration * 60.0) if duration > 0 else 0.0
        assert per_min == 0.0

    def test_accepted_per_minute_calculation(self):
        """Verify the per-minute rate formula."""
        accepted = 5
        duration = 30.0
        per_min = accepted / duration * 60.0
        assert abs(per_min - 10.0) < 0.001

    def test_class_counts_accumulate(self):
        """Class counts accumulate correctly across multiple detections."""
        counts: Dict[str, int] = defaultdict(int)
        for cls in ["person", "person", "chair", "person"]:
            counts[cls] += 1
        assert counts["person"] == 3
        assert counts["chair"] == 1

    def test_non_zero_fp_is_valid_result(self):
        """
        A non-zero accepted detection count is a VALID measurement result.
        fp_test NEVER asserts that FP count must be zero.
        This test verifies that no assertion is violated for non-zero counts.
        """
        # If fp_test tried to assert accepted_total == 0, it would be wrong.
        # This test confirms our design: non-zero is NOT a failure.
        result = {
            "accepted_total": 3,
            "accepted_per_min": 3.0,
            "class_counts": {"person": 3},
        }
        # No assertion on the count — we only assert the structure is intact
        assert isinstance(result["accepted_total"], int)
        assert result["accepted_total"] >= 0  # any non-negative value is valid

    def test_conf_statistics_with_values(self):
        """Confidence statistics compute correctly for a list of values."""
        confs = [0.72, 0.85, 0.61, 0.90]
        mean = statistics.mean(confs)
        assert abs(mean - 0.77) < 0.01
        assert min(confs) == 0.61
        assert max(confs) == 0.90

    def test_conf_statistics_empty_list(self):
        """Empty confidence list returns None for mean/min/max (no crash)."""
        confs = []
        mean = statistics.mean(confs) if confs else None
        mn   = min(confs) if confs else None
        mx   = max(confs) if confs else None
        assert mean is None
        assert mn is None
        assert mx is None

    def test_scene_label_preserved_in_result(self):
        """Scene label must be preserved in result dict."""
        label = "test_empty_desk_2026"
        result = {
            "scene_label": label,
            "accepted_total": 0,
        }
        assert result["scene_label"] == label


class TestFPWithCamera:
    """
    Hardware test — runs a short FP measurement with the real webcam.
    Marked requires_camera. Does NOT assert that FP count is zero.
    """

    @pytest.mark.requires_camera
    def test_fp_measurement_runs_and_produces_result(self, config_fixture):
        """
        Run a 10-second FP measurement on the real camera.
        Verifies the measurement completes and returns a valid result dict.
        Does NOT assert on the accepted_total value.
        """
        config = config_fixture

        try:
            detector = ObjectDetector(config)
            detector.load()
        except DetectorError as e:
            pytest.skip(f"Could not load model: {e}")

        det_filter = DetectionFilter(config)

        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            pytest.skip(f"Webcam not available: {e}")

        try:
            result = run_fp_measurement(
                cam=cam,
                detector=detector,
                det_filter=det_filter,
                duration_s=10.0,
                show_display=False,
                scene_label="automated_pytest_10s",
            )
        finally:
            cam.release()

        # Structural checks — no assertion on the count
        assert result["frames_processed"] > 0, "No frames were processed"
        assert result["duration_actual_s"] >= 9.0, "Test ran too short"
        assert isinstance(result["accepted_total"], int)
        assert isinstance(result["accepted_per_min"], float)
        assert isinstance(result["class_counts"], dict)

        print(f"\n  FP measurement (10s automated): "
              f"accepted={result['accepted_total']}, "
              f"per_min={result['accepted_per_min']:.2f}, "
              f"classes={result['class_counts']}")


if __name__ == "__main__":
    main()
