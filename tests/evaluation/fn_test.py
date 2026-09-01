"""
Empty-Scene Sanity Test — Phase 4
====================================
Filename: fn_test.py
(Name retained for project compatibility; the test is NOT a false-negative test.)

PURPOSE CLARIFICATION — PLEASE READ:
    This script was originally planned as a false-negative (missed detection)
    test. However, false-negative measurement requires a KNOWN OBJECT that
    SHOULD have been detected.

    A blank/empty frame does NOT provide that:
      - Blank frame + zero detections = correct empty-scene behaviour
      - Blank frame + zero detections ≠ evidence of missed detections

    Therefore this file implements an EMPTY-SCENE SANITY TEST:

        Point the camera at an empty scene (or run with a blank frame).
        Count accepted detections.
        Zero accepted detections = correct behaviour in an empty scene.
        Accepted detections in an empty scene = observed false positive events.

    FORMAL FALSE-NEGATIVE MEASUREMENT IS NOT POSSIBLE with this automated
    blank-frame approach because no ground-truth labelled object is present.
    If a labelled sample image is available in data/samples/, this script
    can use it — but only if the ground truth class is known.

    This limitation is documented here and in docs/evaluation.md.

Usage:
    python tests/evaluation/fn_test.py
    python tests/evaluation/fn_test.py --frames 50
    python tests/evaluation/fn_test.py --sample data/samples/person.jpg
    python tests/evaluation/fn_test.py --help

Automated pytest run:
    .venv/Scripts/python.exe -m pytest tests/evaluation/fn_test.py -v -s
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List
from collections import defaultdict

# Make src/ importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import cv2
import numpy as np
import pytest

from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.utils.logger import get_logger
from assistive_navigation.camera.capture import CameraCapture, CameraError
from assistive_navigation.detection.detector import (
    ObjectDetector, Detection, DetectionResult, DetectorError
)
from assistive_navigation.detection.filter import (
    DetectionFilter, FilterResult, RejectionReason
)

logger = get_logger(__name__, level="INFO")


# ─────────────────────────────────────────────────────────────────────────────
# Empty-scene sanity test logic
# ─────────────────────────────────────────────────────────────────────────────
def run_empty_scene_sanity(
    detector: ObjectDetector,
    det_filter: DetectionFilter,
    n_frames: int,
    frame_source: str = "blank",
    cam: Optional[CameraCapture] = None,
) -> dict:
    """
    Run N inferences on blank frames or camera frames.
    Count accepted detections.
    Return a structured result.

    This is NOT a false-negative test.
    Zero accepted detections on blank frames = correct empty-scene behaviour.
    Non-zero accepted detections = observed false positives in that scene.

    Args:
        detector:       Loaded ObjectDetector.
        det_filter:     DetectionFilter.
        n_frames:       Number of frames to process.
        frame_source:   "blank" to use synthesized blank frames,
                        "camera" to use live camera frames.
        cam:            Required if frame_source == "camera".

    Returns:
        dict with fields:
          frame_source         : str
          n_frames_requested   : int
          n_frames_processed   : int
          accepted_total       : int
          raw_total            : int
          accepted_class_counts: dict
          rejected_reason_counts: dict
          is_fn_measurement    : False (always — documents the limitation)
          fn_not_possible_reason: str explaining why
          notes                : list of str
    """
    accepted_total = 0
    raw_total = 0
    accepted_classes: dict = defaultdict(int)
    rejected_reasons: dict = defaultdict(int)
    n_processed = 0

    for i in range(n_frames):
        if frame_source == "blank":
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        elif frame_source == "camera":
            if cam is None:
                break
            frame = cam.read()
            if frame is None:
                logger.warning("Null frame at %d, skipping.", i)
                continue
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

        det_result = detector.detect(frame)
        filtered   = det_filter.filter(det_result)

        raw_total += det_result.raw_count
        accepted_total += filtered.accepted_count
        n_processed += 1

        for det in filtered.accepted:
            accepted_classes[det.class_name] += 1
        for r in filtered.rejected:
            rejected_reasons[r["reason"]] += 1

    notes = []
    if frame_source == "blank":
        notes.append(
            "Blank (all-zero) frames used. "
            "These contain no objects by definition."
        )
        notes.append(
            "Zero accepted detections = expected correct behaviour. "
            "Non-zero = model hallucination on blank input."
        )
        notes.append(
            "This is NOT a false-negative test. "
            "FN measurement requires a known object present in frame."
        )

    return {
        "frame_source":           frame_source,
        "n_frames_requested":     n_frames,
        "n_frames_processed":     n_processed,
        "accepted_total":         accepted_total,
        "raw_total":              raw_total,
        "accepted_class_counts":  dict(accepted_classes),
        "rejected_reason_counts": dict(rejected_reasons),
        "is_fn_measurement":      False,
        "fn_not_possible_reason": (
            "No ground-truth labelled object is present. "
            "FN measurement requires a known object in the frame "
            "that should have been detected. "
            "A blank or empty scene test cannot provide that."
        ),
        "notes": notes,
    }


def run_sample_image_test(
    detector: ObjectDetector,
    det_filter: DetectionFilter,
    image_path: Path,
    expected_class: str,
    n_runs: int = 30,
) -> dict:
    """
    Run N inferences on a saved sample image with a known expected class.

    This is the ONLY mode where this script approaches meaningful
    missed-detection measurement — and only if the sample image genuinely
    contains the expected object.

    The user is responsible for confirming the expected class matches
    the image content. This script does not verify image contents.

    Args:
        detector:        Loaded ObjectDetector.
        det_filter:      DetectionFilter.
        image_path:      Path to the sample image.
        expected_class:  The COCO class name that should be detected.
        n_runs:          Number of times to run inference on the same image.

    Returns:
        dict with measurement results including correct/wrong/no-detection counts.
    """
    if not image_path.exists():
        return {
            "error": f"Sample image not found: {image_path}",
            "is_fn_measurement": False,
        }

    frame = cv2.imread(str(image_path))
    if frame is None:
        return {
            "error": f"Could not read image: {image_path}",
            "is_fn_measurement": False,
        }

    correct_count  = 0
    wrong_count    = 0
    no_det_count   = 0
    accepted_classes: dict = defaultdict(int)
    confidences: List[float] = []

    for _ in range(n_runs):
        det_result = detector.detect(frame)
        filtered   = det_filter.filter(det_result)

        accepted_names = [d.class_name for d in filtered.accepted]

        if expected_class in accepted_names:
            correct_count += 1
            for det in filtered.accepted:
                if det.class_name == expected_class:
                    confidences.append(det.confidence)
        elif len(filtered.accepted) > 0:
            wrong_count += 1
        else:
            no_det_count += 1

        for det in filtered.accepted:
            accepted_classes[det.class_name] += 1

    detection_rate = correct_count / n_runs if n_runs > 0 else 0.0

    return {
        "frame_source":            str(image_path),
        "expected_class":          expected_class,
        "n_runs":                  n_runs,
        "correct_class_count":     correct_count,
        "wrong_class_count":       wrong_count,
        "no_detection_count":      no_det_count,
        "observed_detection_rate": round(detection_rate, 3),
        "accepted_class_counts":   dict(accepted_classes),
        "conf_values":             confidences,
        "conf_mean":               sum(confidences) / len(confidences) if confidences else None,
        "is_fn_measurement":       True,
        "fn_note": (
            "This measurement uses a single static image repeated N times. "
            "It measures detection consistency on that specific image, "
            "not generalisable miss-rate across diverse scenes."
        ),
    }


def print_sanity_result(result: dict) -> None:
    """Print empty-scene sanity or sample-image result."""
    print()
    print("=" * 60)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        print("=" * 60)
        return

    if result.get("expected_class"):
        print("  SAMPLE IMAGE DETECTION TEST")
        print("=" * 60)
        print(f"  Image          : {result['frame_source']}")
        print(f"  Expected class : {result['expected_class']}")
        print(f"  Runs           : {result['n_runs']}")
        print()
        print(f"  Correct-class observations : {result['correct_class_count']}")
        print(f"  Wrong-class observations   : {result['wrong_class_count']}")
        print(f"  No-detection observations  : {result['no_detection_count']}")
        print(f"  Observed detection rate    : {result['observed_detection_rate']:.1%}")
        if result["conf_values"]:
            print(f"  Confidence mean  : {result['conf_mean']:.3f}")
        print()
        print(f"  NOTE: {result['fn_note']}")
    else:
        print("  EMPTY-SCENE SANITY TEST")
        print("=" * 60)
        print(f"  Frame source   : {result['frame_source']}")
        print(f"  Frames processed: {result['n_frames_processed']}")
        print()
        print(f"  Raw detections total    : {result['raw_total']}")
        print(f"  Accepted detections     : {result['accepted_total']}")
        if result["accepted_class_counts"]:
            print(f"  Accepted classes        : {result['accepted_class_counts']}")
        else:
            print(f"  Accepted classes        : NONE")
        print()
        for note in result.get("notes", []):
            print(f"  NOTE: {note}")
        print()
        print(f"  IS THIS A FALSE-NEGATIVE TEST? {result['is_fn_measurement']}")
        print(f"  WHY NOT: {result['fn_not_possible_reason']}")

    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Empty-Scene Sanity Test (NOT a FN test)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: This script does NOT measure false negatives unless a
      known-object sample image is provided via --sample.

Examples:
  python tests/evaluation/fn_test.py
  python tests/evaluation/fn_test.py --frames 60
  python tests/evaluation/fn_test.py --camera
  python tests/evaluation/fn_test.py --sample data/samples/person.jpg --expected-class person
        """
    )
    parser.add_argument(
        "--frames", type=int, default=30,
        help="Number of frames to process (default: 30)"
    )
    parser.add_argument(
        "--camera", action="store_true",
        help="Use live camera instead of blank frames"
    )
    parser.add_argument(
        "--sample", type=str, default="",
        help="Path to a sample image with a known object"
    )
    parser.add_argument(
        "--expected-class", type=str, default="",
        help="Expected COCO class name in the sample image (e.g. 'person')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()

    print()
    print("=" * 60)
    print("  PHASE 4 — EMPTY-SCENE SANITY TEST")
    print("  (NOT a false-negative measurement — see docstring)")
    print("=" * 60)

    # Load model
    try:
        detector = ObjectDetector(config)
        detector.load()
        print(f"  Model loaded: {detector.model_name}")
    except DetectorError as e:
        print(f"  ERROR loading model: {e}")
        sys.exit(1)

    det_filter = DetectionFilter(config)

    # Sample image mode
    if args.sample:
        sample_path = Path(args.sample)
        if not args.expected_class:
            print("  ERROR: --expected-class is required when using --sample")
            sys.exit(1)
        print(f"\n  Running sample image test:")
        print(f"    Image:          {sample_path}")
        print(f"    Expected class: {args.expected_class}")
        print(f"    Runs:           {args.frames}")
        result = run_sample_image_test(
            detector, det_filter, sample_path,
            args.expected_class, n_runs=args.frames
        )
        print_sanity_result(result)
        return

    # Camera mode
    if args.camera:
        print("\n  Using live camera for empty-scene test.")
        print("  Point the camera at an empty scene (no navigation objects).")
        input("  Press ENTER when ready ...")
        try:
            cam = CameraCapture(config)
            cam.open()
        except CameraError as e:
            print(f"  ERROR: Camera problem: {e}")
            sys.exit(1)
        try:
            result = run_empty_scene_sanity(
                detector, det_filter, args.frames,
                frame_source="camera", cam=cam
            )
        finally:
            cam.release()
        print_sanity_result(result)
        return

    # Default: blank frames
    print(f"\n  Running blank-frame sanity test ({args.frames} frames) ...")
    result = run_empty_scene_sanity(
        detector, det_filter, args.frames, frame_source="blank"
    )
    print_sanity_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# Automated pytest tests — no camera required, no manual interaction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def loaded_detector_fn():
    """Load detector once for the fn_test module."""
    config = load_config()
    detector = ObjectDetector(config)
    try:
        detector.load()
    except DetectorError as e:
        pytest.skip(f"Could not load YOLO model: {e}")
    return detector


@pytest.fixture(scope="module")
def fn_filter():
    """DetectionFilter for fn_test module."""
    config = load_config()
    return DetectionFilter(config)


class TestEmptySceneSanity:
    """
    Tests for empty-scene behaviour.
    These run with blank (all-zero) frames — no webcam required.
    """

    def test_blank_frame_produces_zero_raw_detections(
        self, loaded_detector_fn
    ):
        """
        A completely black frame should produce zero raw detections.
        Tests that the model does not hallucinate on blank input.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = loaded_detector_fn.detect(frame)
        assert result.raw_count == 0, (
            f"Detected {result.raw_count} objects in a blank frame. "
            f"Classes: {[d.class_name for d in result.detections]}. "
            f"This indicates model hallucination on blank input."
        )

    def test_blank_frame_produces_zero_accepted(
        self, loaded_detector_fn, fn_filter
    ):
        """
        Blank frame → no raw detections → no accepted detections.
        Verifies the pipeline handles empty detection results correctly.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det_result = loaded_detector_fn.detect(frame)
        filtered   = fn_filter.filter(det_result)
        assert filtered.accepted_count == 0, (
            f"Expected 0 accepted on blank frame, got {filtered.accepted_count}. "
            f"Accepted: {[d.class_name for d in filtered.accepted]}"
        )

    def test_run_empty_scene_sanity_returns_structure(
        self, loaded_detector_fn, fn_filter
    ):
        """run_empty_scene_sanity returns a dict with required keys."""
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=5, frame_source="blank"
        )
        required_keys = [
            "frame_source", "n_frames_requested", "n_frames_processed",
            "accepted_total", "raw_total", "accepted_class_counts",
            "rejected_reason_counts", "is_fn_measurement",
            "fn_not_possible_reason", "notes"
        ]
        for key in required_keys:
            assert key in result, f"Missing key in result: {key}"

    def test_blank_sanity_is_not_fn_measurement(
        self, loaded_detector_fn, fn_filter
    ):
        """
        is_fn_measurement must be False for blank-frame tests.
        This enforces the documentation requirement: do not misrepresent
        an empty-scene test as a false-negative measurement.
        """
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=3, frame_source="blank"
        )
        assert result["is_fn_measurement"] is False, (
            "Blank-frame test must NOT be labelled as an FN measurement. "
            "Set is_fn_measurement=False for empty-scene sanity tests."
        )

    def test_blank_sanity_has_fn_explanation(
        self, loaded_detector_fn, fn_filter
    ):
        """fn_not_possible_reason must be a non-empty string."""
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=3, frame_source="blank"
        )
        assert isinstance(result["fn_not_possible_reason"], str)
        assert len(result["fn_not_possible_reason"]) > 20, (
            "fn_not_possible_reason must contain a meaningful explanation."
        )

    def test_blank_sanity_zero_accepted_30_frames(
        self, loaded_detector_fn, fn_filter
    ):
        """
        Run 30 blank frames. Expect zero accepted detections.
        This is the main quality gate for blank-frame behaviour.
        """
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=30, frame_source="blank"
        )
        assert result["accepted_total"] == 0, (
            f"Expected 0 accepted detections across 30 blank frames, "
            f"got {result['accepted_total']}. "
            f"Classes: {result['accepted_class_counts']}. "
            f"This indicates model is hallucinating on blank frames."
        )

    def test_frames_processed_matches_requested(
        self, loaded_detector_fn, fn_filter
    ):
        """All requested frames should be processed (blank frame always available)."""
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=10, frame_source="blank"
        )
        assert result["n_frames_processed"] == 10

    def test_notes_list_is_present_for_blank(
        self, loaded_detector_fn, fn_filter
    ):
        """Notes list must be present and non-empty for blank-frame tests."""
        result = run_empty_scene_sanity(
            loaded_detector_fn, fn_filter,
            n_frames=3, frame_source="blank"
        )
        assert isinstance(result["notes"], list)
        assert len(result["notes"]) > 0, "Notes list should have at least one entry."


class TestSampleImageTest:
    """
    Tests for sample-image based detection (the only mode that can
    approach FN measurement). Most tests run with synthetic paths.
    """

    def test_missing_sample_returns_error_dict(
        self, loaded_detector_fn, fn_filter
    ):
        """
        If the sample image does not exist, return error dict — do not crash.
        """
        result = run_sample_image_test(
            loaded_detector_fn, fn_filter,
            image_path=Path("nonexistent_image_abc123.jpg"),
            expected_class="person",
            n_runs=1,
        )
        assert "error" in result, "Should return error dict for missing image"

    def test_sample_result_is_fn_measurement(
        self, loaded_detector_fn, fn_filter, tmp_path
    ):
        """
        If a real image is provided, is_fn_measurement must be True.
        (Testing with a minimal synthetic image that is valid but blank)
        """
        # Create a minimal valid image file
        test_img = tmp_path / "test_img.jpg"
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(test_img), blank)

        result = run_sample_image_test(
            loaded_detector_fn, fn_filter,
            image_path=test_img,
            expected_class="person",
            n_runs=3,
        )
        # Should succeed in loading (blank, even if detects nothing)
        assert "error" not in result
        assert result["is_fn_measurement"] is True


if __name__ == "__main__":
    main()
