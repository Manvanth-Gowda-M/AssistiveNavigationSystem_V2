"""
Assistive Navigation System V2 — Main Entry Point
===================================================
Phase 13: Full pipeline integration.

Run:
    python -m assistive_navigation
    python -m assistive_navigation --headless
    python -m assistive_navigation --no-depth
    python -m assistive_navigation --help

SAFETY NOTICE:
    This is a research prototype. It must NOT be used as a substitute
    for a mobility aid, white cane, guide dog, or human assistance.
    Detection quality has known limitations (see docs/evaluation.md).
    Phase 15 empirical calibration is still required.

Pipeline stages (in order each frame):
  1. CameraCapture         → BGR frame
  2. ObjectDetector        → DetectionResult
  3. DetectionFilter       → FilterResult
  4. ObjectTracker         → List[TrackedObject]
  5. DepthEstimator        → Optional[depth_map]   (every N frames)
  6. DepthFusion           → List[FusedObject]
  7. SpatialReasoner       → List[DirectedObject]
  8. NavigationPriorityEngine → List[ScoredObject]
  9. TemporalConfirmationFilter → List[ConfirmedObject]
  10. AlertManager         → AlertResult → SimpleAudioQueue → TTS

Depth update interval:
    Depth inference is expensive (~120-370ms on CPU). It runs every
    depth_update_interval frames (default: 5). Between updates the
    cached depth map is reused and depth_age increments. This is
    preserved exactly from Phase 6.
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ── ensure src/ is importable when run as a script ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.utils.logger        import get_logger
from assistive_navigation.utils.perf_monitor  import PerfMonitor

from assistive_navigation.camera.capture      import CameraCapture, CameraError
from assistive_navigation.detection.detector  import ObjectDetector, DetectorError
from assistive_navigation.detection.filter    import DetectionFilter
from assistive_navigation.tracking.tracker    import ObjectTracker
from assistive_navigation.depth.depth_estimator import DepthEstimator, DepthEstimatorError
from assistive_navigation.depth.fusion        import DepthFusion
from assistive_navigation.navigation.spatial  import SpatialReasoner
from assistive_navigation.navigation.priority import NavigationPriorityEngine
from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
from assistive_navigation.alerts.alert_manager import AlertManager
from assistive_navigation.audio.queue         import SimpleAudioQueue
from assistive_navigation.audio.tts           import TextToSpeech
from assistive_navigation.visualization.debug_view import DebugView

logger = get_logger(__name__, level="INFO")


# ---------------------------------------------------------------------------
# FrameResult — per-frame summary returned by run_frame()
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """
    Summary of one pipeline frame.
    Returned by run_frame() for testing, logging, and monitoring.
    """
    frame_num:           int
    camera_ok:           bool
    detections_raw:      int
    detections_accepted: int
    tracks_active:       int
    depth_available:     bool
    depth_ran:           bool     # True if depth inference actually ran this frame
    depth_age:           int
    fused_count:         int
    confirmed_count:     int
    alert_issued:        bool
    alert_text:          str
    detect_latency_ms:   float
    depth_latency_ms:    float
    total_latency_ms:    float
    pipeline_fps:        float
    error:               Optional[str] = None


# ---------------------------------------------------------------------------
# AssistiveNavigationPipeline
# ---------------------------------------------------------------------------

class AssistiveNavigationPipeline:
    """
    Orchestrates all pipeline stages from camera capture to speech output.

    Lifecycle:
        pipeline = AssistiveNavigationPipeline(config)
        ok = pipeline.start()       # load models, open camera, init audio
        if not ok:
            sys.exit(1)
        pipeline.run()              # main loop — blocks until Ctrl+C
        pipeline.stop()             # called automatically by run() on exit

    Or for frame-by-frame control (testing):
        pipeline.start()
        result = pipeline.run_frame()   # one frame
        pipeline.stop()

    SAFETY NOTE:
        Research prototype only. Not validated for independent navigation.
        Known Phase 4 false positives may trigger spurious speech.
        Phase 15 calibration required before navigation use.
    """

    def __init__(self, config: dict, headless: bool = False) -> None:
        self._config   = config
        self._headless = headless
        self._running  = False

        pip_cfg = config.get("pipeline", {})
        self._max_errors = int(pip_cfg.get("max_consecutive_errors", 10))
        show_window = bool(pip_cfg.get("show_debug_window", True))
        if headless:
            show_window = False

        # Instantiate all components (models not loaded yet)
        self._cam     = CameraCapture(config)
        self._detector= ObjectDetector(config)
        self._filter  = DetectionFilter(config)
        self._tracker = ObjectTracker(config)
        self._depth   = DepthEstimator(config)
        self._fusion  = DepthFusion(config)
        self._spatial = SpatialReasoner(config)
        self._priority= NavigationPriorityEngine(config)
        self._temporal= TemporalConfirmationFilter(config)

        self._tts   = TextToSpeech(config)
        self._queue = SimpleAudioQueue(self._tts)
        self._alert = AlertManager(config, audio_queue=self._queue)

        self._debug = DebugView(config)
        if headless:
            self._debug._show_window = False

        self._perf = PerfMonitor(window=30)

        # Track last alert for display
        self._last_alert_result = None

        logger.debug("AssistiveNavigationPipeline created (headless=%s)", headless)

    # -----------------------------------------------------------------------
    # Startup
    # -----------------------------------------------------------------------

    def start(self) -> bool:
        """
        Load all models, open camera, initialise audio.

        Returns True if all critical components are ready.
        Returns False if a critical component failed (camera or detector).
        Non-critical failures (depth, TTS) print a warning and continue.

        Prints a formatted startup status to stdout.
        """
        self._print_header()

        ok = True

        # TTS and queue (non-critical)
        tts_ok = self._tts.initialise()
        if tts_ok:
            self._queue.start()
            self._status_line("Audio", f"SAPI5 / {self._tts.voice_name}", ok=True)
        else:
            self._status_line("Audio", "unavailable — speech disabled", ok=False, critical=False)

        # Detector (critical)
        try:
            self._detector.load()
            self._status_line(
                "Detector",
                f"{self._detector.model_name} (CPU)",
                ok=True
            )
        except DetectorError as e:
            self._status_line("Detector", f"FAILED: {e}", ok=False, critical=True)
            ok = False

        # Depth estimator (non-critical — pipeline degrades gracefully)
        if ok:
            try:
                self._depth.load()
                self._status_line(
                    "Depth",
                    f"DA-V2-Small ({self._depth.input_size}px, "
                    f"every {self._depth.update_interval} frames)",
                    ok=True
                )
            except DepthEstimatorError as e:
                self._status_line("Depth", f"unavailable — {e}", ok=False, critical=False)

        # Camera (critical)
        if ok:
            try:
                self._cam.open()
                self._status_line(
                    "Camera",
                    f"{self._cam.actual_width}×{self._cam.actual_height} (device {self._cam.device_id})",
                    ok=True
                )
            except CameraError as e:
                self._status_line("Camera", f"FAILED: {e}", ok=False, critical=True)
                ok = False

        if ok:
            self._status_line("Tracker",  "ByteTrack (CPU)", ok=True)
            print("=" * 60)
            print("  SYSTEM READY")
            print("  Research prototype — NOT a mobility aid replacement.")
            if not self._headless:
                print("  Press Q in the debug window, or Ctrl+C, to exit.")
            else:
                print("  Press Ctrl+C to exit.")
            print("=" * 60)
        else:
            print("=" * 60)
            print("  STARTUP FAILED — check errors above.")
            print("=" * 60)

        self._running = ok
        return ok

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> None:
        """
        Main loop: call run_frame() continuously until stopped.

        Handles KeyboardInterrupt and consecutive errors gracefully.
        Calls stop() on exit.
        """
        consecutive_errors = 0

        try:
            while self._running:
                try:
                    result = self.run_frame()
                    consecutive_errors = 0

                    # Q key in debug window → clean stop
                    if result.error == "quit_requested":
                        logger.info("User quit via debug window.")
                        break

                except Exception as e:
                    consecutive_errors += 1
                    logger.warning(
                        "Frame %d error (%d/%d): %s",
                        self._perf.frame_count, consecutive_errors,
                        self._max_errors, e
                    )
                    if consecutive_errors >= self._max_errors:
                        logger.error(
                            "Too many consecutive errors (%d). Stopping pipeline.",
                            self._max_errors
                        )
                        break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping pipeline.")
        finally:
            self.stop()

    # -----------------------------------------------------------------------
    # Single frame
    # -----------------------------------------------------------------------

    def run_frame(self) -> FrameResult:
        """
        Process one frame through the complete pipeline.

        Depth estimation respects depth_update_interval — inference only
        runs every N frames; between updates the cached depth map and
        incremented depth_age are used automatically (Phase 6 behaviour
        preserved exactly via DepthEstimator.process_frame()).

        Returns FrameResult with per-stage stats.
        Never raises — all stage exceptions are caught and logged.
        """
        self._perf.start_frame()
        t_frame_start = time.perf_counter()

        # ── 1. Camera ───────────────────────────────────────────────────────
        frame = self._cam.read()
        if frame is None:
            logger.warning("Camera returned None frame.")
            result = FrameResult(
                frame_num=self._perf.frame_count,
                camera_ok=False,
                detections_raw=0, detections_accepted=0,
                tracks_active=0, depth_available=False,
                depth_ran=False, depth_age=0,
                fused_count=0, confirmed_count=0,
                alert_issued=False, alert_text="",
                detect_latency_ms=0, depth_latency_ms=0,
                total_latency_ms=0, pipeline_fps=0,
                error="camera_none",
            )
            self._perf.end_frame()
            return result

        frame_h, frame_w = frame.shape[:2]

        # ── 2. Detection ─────────────────────────────────────────────────────
        det_result   = self._safe_detect(frame)
        detect_ms    = det_result.latency_ms if det_result else 0.0
        raw_count    = det_result.raw_count  if det_result else 0

        # ── 3. Filter ────────────────────────────────────────────────────────
        filtered = self._filter.filter(det_result) if det_result else None
        accepted = filtered.accepted if filtered else []
        acc_count = len(accepted)

        # ── 4. Tracking ──────────────────────────────────────────────────────
        tracked = []
        try:
            tracked = self._tracker.update(accepted, frame_w, frame_h)
        except Exception as e:
            logger.warning("Tracker error: %s", e)

        # ── 5. Depth estimation ──────────────────────────────────────────────
        # process_frame() handles the depth_update_interval internally.
        # It runs inference every N frames and caches the result.
        # depth_age = 0 when inference just ran; >0 when using cached map.
        depth_map   = None
        depth_ms    = 0.0
        depth_ran   = False
        depth_age   = 0

        if self._depth.is_loaded:
            try:
                prev_count = self._depth.inference_count
                depth_map  = self._depth.process_frame(frame)
                depth_ran  = (self._depth.inference_count > prev_count)
                depth_ms   = self._depth.last_latency_ms if depth_ran else 0.0
                depth_age  = self._depth.depth_frame_age
            except Exception as e:
                logger.warning("Depth error: %s", e)
                depth_map = None

        depth_available = depth_map is not None

        # ── 6. Fusion ────────────────────────────────────────────────────────
        fused = []
        try:
            fused = self._fusion.fuse(tracked, depth_map, depth_age)
        except Exception as e:
            logger.warning("Fusion error: %s", e)

        # ── 7. Spatial direction ─────────────────────────────────────────────
        directed = []
        try:
            directed = self._spatial.assign_direction(fused)
        except Exception as e:
            logger.warning("Spatial error: %s", e)

        # ── 8. Priority scoring ──────────────────────────────────────────────
        scored = []
        try:
            scored = self._priority.score(directed)
        except Exception as e:
            logger.warning("Priority error: %s", e)

        # ── 9. Temporal confirmation ─────────────────────────────────────────
        confirmed = []
        try:
            confirmed = self._temporal.update(scored)
        except Exception as e:
            logger.warning("Temporal error: %s", e)

        confirmed_count = len(confirmed)

        # ── 10. Alert manager ────────────────────────────────────────────────
        alert_result = None
        try:
            alert_result = self._alert.process(confirmed)
            if alert_result.alert_issued:
                self._last_alert_result = alert_result
                logger.info(
                    "Alert: %r (reason=%s, track=%d)",
                    alert_result.alert_text,
                    alert_result.trigger_reason,
                    alert_result.selected_track,
                )
        except Exception as e:
            logger.warning("Alert manager error: %s", e)

        alert_issued = bool(alert_result and alert_result.alert_issued)
        alert_text   = alert_result.alert_text if alert_result else ""

        # ── 11. Debug visualization ──────────────────────────────────────────
        quit_requested = False
        if self._debug.is_enabled:
            try:
                vis = self._debug.draw(
                    frame, confirmed,
                    alert_result or self._last_alert_result,
                    self._perf
                )
                quit_requested = self._debug.show(vis)
            except Exception as e:
                logger.debug("Debug view error (non-critical): %s", e)

        # ── Performance recording ────────────────────────────────────────────
        total_ms = (time.perf_counter() - t_frame_start) * 1000.0
        self._perf.record(
            detect_ms=detect_ms,
            depth_ms=depth_ms,
            depth_ran=depth_ran,
            detections_raw=raw_count,
            detections_acc=acc_count,
            tracks_active=len(tracked),
            confirmed_count=confirmed_count,
            alert_issued=alert_issued,
        )
        stats = self._perf.end_frame()

        # Periodic console log
        if self._perf.frame_count % 30 == 0:
            logger.info(
                "Frame %d | %s | depth_age=%d%s",
                self._perf.frame_count,
                self._perf.summary_line(),
                depth_age,
                " [depth ran]" if depth_ran else "",
            )

        return FrameResult(
            frame_num=self._perf.frame_count,
            camera_ok=True,
            detections_raw=raw_count,
            detections_accepted=acc_count,
            tracks_active=len(tracked),
            depth_available=depth_available,
            depth_ran=depth_ran,
            depth_age=depth_age,
            fused_count=len(fused),
            confirmed_count=confirmed_count,
            alert_issued=alert_issued,
            alert_text=alert_text,
            detect_latency_ms=detect_ms,
            depth_latency_ms=depth_ms,
            total_latency_ms=total_ms,
            pipeline_fps=self._perf.fps,
            error="quit_requested" if quit_requested else None,
        )

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    def stop(self) -> None:
        """
        Shut down all pipeline components cleanly.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if not self._running and not self._cam.is_open:
            return

        logger.info("Pipeline shutting down ...")
        self._running = False

        try:
            self._queue.stop()
        except Exception as e:
            logger.debug("Queue stop error: %s", e)

        try:
            self._tts.shutdown()
        except Exception as e:
            logger.debug("TTS shutdown error: %s", e)

        try:
            self._cam.release()
        except Exception as e:
            logger.debug("Camera release error: %s", e)

        try:
            self._debug.close()
        except Exception as e:
            logger.debug("Debug view close error: %s", e)

        logger.info("Pipeline stopped cleanly.")
        print("\nSystem stopped cleanly.")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _safe_detect(self, frame):
        """Run detection; return None on error."""
        try:
            return self._detector.detect(frame)
        except Exception as e:
            logger.warning("Detection error: %s", e)
            return None

    @staticmethod
    def _print_header() -> None:
        print()
        print("=" * 60)
        print("  ASSISTIVE NAVIGATION SYSTEM V2")
        print("  Research Prototype — Phase 13 Integration")
        print("=" * 60)
        print("  SYSTEM STARTING ...")
        print()

    @staticmethod
    def _status_line(
        component: str, detail: str,
        ok: bool, critical: bool = True
    ) -> None:
        flag = "[OK ]" if ok else ("[WARN]" if not critical else "[FAIL]")
        print(f"  {flag} {component:<12}: {detail}")

    def __repr__(self) -> str:
        return (
            f"AssistiveNavigationPipeline("
            f"running={self._running}, "
            f"frames={self._perf.frame_count}, "
            f"fps={self._perf.fps:.1f})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Assistive Navigation System V2 — Research Prototype",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SAFETY NOTICE:
  This is a research/prototype system. It must NOT be used as a
  substitute for a mobility aid, cane, guide dog, or human assistance.
        """
    )
    p.add_argument(
        "--headless", action="store_true",
        help="Disable the OpenCV debug window (audio-only mode)"
    )
    p.add_argument(
        "--config", type=str, default=None,
        help="Path to a custom config.yaml (default: config/config.yaml)"
    )
    return p.parse_args()


def main() -> None:
    """Entry point for `python -m assistive_navigation`."""
    args = parse_args()

    # Load config
    from assistive_navigation.utils.config_loader import load_config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Could not load config: {e}")
        sys.exit(1)

    pipeline = AssistiveNavigationPipeline(config, headless=args.headless)

    if not pipeline.start():
        sys.exit(1)

    pipeline.run()   # blocks until Ctrl+C or Q


if __name__ == "__main__":
    main()
