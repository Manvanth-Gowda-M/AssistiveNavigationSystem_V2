"""
Detector Benchmark Tool
========================
Phase 3 evaluation script.

Runs YOLO11n + DetectionFilter on live webcam frames and records:
  - Per-frame inference latency
  - Pipeline FPS (camera capture + detection)
  - Distribution of detected classes (raw and filtered)
  - All raw detections including rejected ones with reasons

Output:
  - Console summary printed at end
  - logs/detections.csv  — every detection (raw + filter decision)
  - logs/benchmark.csv   — per-frame latency/FPS

Usage (from project root with venv activated):
    python tools/benchmark_detector.py
    python tools/benchmark_detector.py --frames 120
    python tools/benchmark_detector.py --frames 60 --no-display

This script is for Phase 3 manual evaluation.
It does NOT test tracking, depth, or alerts.

Press Q in the display window, or Ctrl+C in the terminal, to stop early.
"""

import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict

# Make sure src/ is on the path so imports work when run from project root
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

import cv2
import numpy as np

from assistive_navigation.utils.config_loader import load_config, get_logs_dir
from assistive_navigation.utils.logger import get_logger
from assistive_navigation.camera.capture import CameraCapture, CameraError
from assistive_navigation.detection.detector import ObjectDetector, DetectorError
from assistive_navigation.detection.filter import DetectionFilter, RejectionReason

logger = get_logger(__name__, level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Detector Benchmark")
    parser.add_argument(
        "--frames", type=int, default=60,
        help="Number of frames to process (default: 60)"
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable the OpenCV debug window (headless mode)"
    )
    return parser.parse_args()


def draw_debug_overlay(frame, result, filtered, latency_ms, fps):
    """
    Draw bounding boxes, class names, confidence, and filter decisions
    on the frame for visual verification.

    Green box  = accepted (navigation-relevant, above confidence threshold)
    Red box    = rejected (ignored class or low confidence)
    Yellow text = raw detection info
    """
    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Draw accepted detections — GREEN
    for det in filtered.accepted:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        priority = filtered.get_priority(det)
        label = f"{det.class_name} {det.confidence:.2f} [{priority[:4]}]"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(overlay, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    # Draw rejected detections — RED (semi-transparent)
    for rej in filtered.rejected:
        x1, y1, x2, y2 = [int(v) for v in rej["bbox"]]
        reason_short = {
            RejectionReason.LOW_CONFIDENCE: "low_conf",
            RejectionReason.CLASS_IGNORED:  "ignored",
            RejectionReason.CLASS_NOT_IN_LIST: "not_in_list"
        }.get(rej["reason"], rej["reason"])
        label = f"{rej['class_name']} {rej['confidence']:.2f} [{reason_short}]"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), 1)
        cv2.putText(overlay, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 200), 1)

    # HUD: FPS and latency
    cv2.putText(overlay,
                f"FPS: {fps:.1f} | Latency: {latency_ms:.0f}ms",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    cv2.putText(overlay,
                f"Raw:{result.raw_count} Acc:{filtered.accepted_count} Rej:{filtered.rejected_count}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(overlay,
                "GREEN=accepted  RED=rejected  Q=quit",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return overlay


def main():
    args = parse_args()
    config = load_config()
    logs_dir = get_logs_dir()
    logs_dir.mkdir(exist_ok=True)

    detection_csv  = logs_dir / "detections.csv"
    benchmark_csv  = logs_dir / "benchmark.csv"

    logger.info("=== Phase 3 Detector Benchmark ===")
    logger.info("Frames to process : %d", args.frames)
    logger.info("Display window    : %s", "disabled" if args.no_display else "enabled")
    logger.info("Detection CSV     : %s", detection_csv)
    logger.info("Benchmark CSV     : %s", benchmark_csv)

    # --- Load model ---
    logger.info("Loading YOLO model ...")
    try:
        detector = ObjectDetector(config)
        detector.load()
        logger.info("Model loaded: %s", detector.model_name)
    except DetectorError as e:
        logger.error("Could not load model: %s", e)
        sys.exit(1)

    det_filter = DetectionFilter(config)

    # --- Open camera ---
    logger.info("Opening camera ...")
    try:
        cam = CameraCapture(config)
        cam.open()
        logger.info("Camera ready: %dx%d", cam.actual_width, cam.actual_height)
    except CameraError as e:
        logger.error("Camera error: %s", e)
        sys.exit(1)

    # --- CSV writers ---
    det_file = open(detection_csv, "w", newline="", encoding="utf-8")
    bench_file = open(benchmark_csv, "w", newline="", encoding="utf-8")

    det_writer = csv.writer(det_file)
    det_writer.writerow([
        "frame_num", "class_id", "class_name", "confidence",
        "x1", "y1", "x2", "y2", "center_x", "center_y",
        "area_norm", "filter_decision", "rejection_reason"
    ])

    bench_writer = csv.writer(bench_file)
    bench_writer.writerow([
        "frame_num", "latency_ms", "raw_count", "accepted_count",
        "rejected_count", "smoothed_fps"
    ])

    # --- Per-class tallies ---
    raw_class_counts    = defaultdict(int)
    accepted_class_counts = defaultdict(int)
    rejected_reasons    = defaultdict(int)
    latencies           = []

    logger.info("Running benchmark ... (press Q in window or Ctrl+C to stop)")

    frame_num = 0
    try:
        while frame_num < args.frames:
            frame = cam.read()
            if frame is None:
                logger.warning("Null frame at frame %d, skipping.", frame_num)
                continue

            result   = detector.detect(frame)
            filtered = det_filter.filter(result)
            fps      = cam.get_smoothed_fps()
            latencies.append(result.latency_ms)

            # --- Write detection CSV ---
            for det in result.detections:
                is_accepted = det in filtered.accepted
                rej_reason  = ""
                if not is_accepted:
                    for r in filtered.rejected:
                        if (r["class_id"] == det.class_id and
                                abs(r["confidence"] - det.confidence) < 0.001):
                            rej_reason = r["reason"]
                            break
                det_writer.writerow([
                    frame_num,
                    det.class_id, det.class_name, f"{det.confidence:.4f}",
                    f"{det.bbox[0]:.1f}", f"{det.bbox[1]:.1f}",
                    f"{det.bbox[2]:.1f}", f"{det.bbox[3]:.1f}",
                    f"{det.center_x:.4f}", f"{det.center_y:.4f}",
                    f"{det.area_norm:.4f}",
                    "accepted" if is_accepted else "rejected",
                    rej_reason
                ])
                raw_class_counts[det.class_name] += 1
                if is_accepted:
                    accepted_class_counts[det.class_name] += 1
                elif rej_reason:
                    rejected_reasons[rej_reason] += 1

            # --- Write benchmark CSV ---
            bench_writer.writerow([
                frame_num, f"{result.latency_ms:.2f}",
                result.raw_count, filtered.accepted_count,
                filtered.rejected_count, f"{fps:.2f}"
            ])

            frame_num += 1

            # --- Debug display ---
            if not args.no_display:
                display_frame = draw_debug_overlay(
                    frame, result, filtered, result.latency_ms, fps
                )
                cv2.imshow("Phase 3 Benchmark — Q to quit", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed Q — stopping early at frame %d.", frame_num)
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted at frame %d.", frame_num)
    finally:
        cam.release()
        det_file.close()
        bench_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    # --- Summary ---
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        est_fps = 1000.0 / avg_lat if avg_lat > 0 else 0

        print("\n" + "=" * 55)
        print("  PHASE 3 BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"  Frames processed    : {frame_num}")
        print(f"  Avg inference latency: {avg_lat:.1f} ms")
        print(f"  Min latency          : {min_lat:.1f} ms")
        print(f"  Max latency          : {max_lat:.1f} ms")
        print(f"  Est. detection FPS   : {est_fps:.1f}")
        print(f"  Performance target   : < 200 ms / > 5 FPS")
        print()
        print(f"  Classes detected (raw, all frames):")
        for cls, count in sorted(raw_class_counts.items(),
                                 key=lambda x: -x[1]):
            accepted = accepted_class_counts.get(cls, 0)
            print(f"    {cls:<20} raw={count:4d}  accepted={accepted:4d}")
        print()
        print(f"  Rejection breakdown:")
        for reason, count in sorted(rejected_reasons.items(),
                                    key=lambda x: -x[1]):
            print(f"    {reason:<25}: {count}")
        print()
        print(f"  CSV output:")
        print(f"    {detection_csv}")
        print(f"    {benchmark_csv}")
        print("=" * 55)
        logger.info("Benchmark complete.")
    else:
        logger.warning("No frames were processed.")


if __name__ == "__main__":
    main()
