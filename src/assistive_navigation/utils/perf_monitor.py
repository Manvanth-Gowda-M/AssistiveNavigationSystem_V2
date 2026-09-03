"""
Performance Monitor — Phase 13 Minimal Implementation
=======================================================
Measures per-frame timing across pipeline stages.

Phase: 13 (minimal — Phase 14 expands to full CPU/RAM metrics)
Status: IMPLEMENTED

What this module measures:
  - Camera capture time
  - Detection latency (from DetectionResult.latency_ms)
  - Depth latency (from DepthEstimator.last_latency_ms; 0 on skipped frames)
  - Total per-frame latency (camera read → alert decision complete)
  - Rolling FPS over last N frames

What Phase 14 will add:
  - CPU usage (psutil)
  - RAM usage (psutil)
  - Per-stage breakdown written to logs/performance.csv

Usage:
    from assistive_navigation.utils.perf_monitor import PerfMonitor

    mon = PerfMonitor()
    mon.start_frame()
    # ... pipeline stages ...
    mon.record(detect_ms=65.0, depth_ms=0.0)
    mon.end_frame()
    print(mon.fps, mon.avg_detect_ms)
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class FrameStats:
    """Timing statistics for one pipeline frame."""
    frame_num:        int   = 0
    total_ms:         float = 0.0    # wall-clock time: start_frame → end_frame
    detect_ms:        float = 0.0    # YOLO inference latency (DetectionResult)
    depth_ms:         float = 0.0    # depth inference latency (0 if skipped)
    depth_ran:        bool  = False  # True if depth inference actually ran
    detections_raw:   int   = 0
    detections_acc:   int   = 0
    tracks_active:    int   = 0
    confirmed_count:  int   = 0
    alert_issued:     bool  = False


class PerfMonitor:
    """
    Lightweight per-frame performance monitor.

    Call start_frame() at the beginning of each pipeline cycle,
    populate the fields via record(), then call end_frame().
    fps and average latencies update automatically.

    Args:
        window: Number of recent frames used for rolling averages (default 30).
    """

    def __init__(self, window: int = 30) -> None:
        self._window   = window
        self._frame_ts: Deque[float] = deque(maxlen=window)   # end-of-frame timestamps
        self._total_ms: Deque[float] = deque(maxlen=window)
        self._det_ms:   Deque[float] = deque(maxlen=window)
        self._dep_ms:   Deque[float] = deque(maxlen=window)

        self._frame_num:  int   = 0
        self._t_frame_start: Optional[float] = None

        # Current frame stats (populated between start/end)
        self.current = FrameStats()

    # -----------------------------------------------------------------------
    # Frame lifecycle
    # -----------------------------------------------------------------------

    def start_frame(self) -> None:
        """Mark the start of a new pipeline frame."""
        self._frame_num += 1
        self._t_frame_start = time.perf_counter()
        self.current = FrameStats(frame_num=self._frame_num)

    def record(
        self,
        detect_ms:       float = 0.0,
        depth_ms:        float = 0.0,
        depth_ran:       bool  = False,
        detections_raw:  int   = 0,
        detections_acc:  int   = 0,
        tracks_active:   int   = 0,
        confirmed_count: int   = 0,
        alert_issued:    bool  = False,
    ) -> None:
        """Populate timing and count fields for the current frame."""
        self.current.detect_ms       = detect_ms
        self.current.depth_ms        = depth_ms
        self.current.depth_ran       = depth_ran
        self.current.detections_raw  = detections_raw
        self.current.detections_acc  = detections_acc
        self.current.tracks_active   = tracks_active
        self.current.confirmed_count = confirmed_count
        self.current.alert_issued    = alert_issued

    def end_frame(self) -> FrameStats:
        """
        Mark the end of the current frame.
        Records total elapsed time and updates rolling averages.
        Returns the completed FrameStats.
        """
        if self._t_frame_start is not None:
            self.current.total_ms = (time.perf_counter() - self._t_frame_start) * 1000.0

        now = time.perf_counter()
        self._frame_ts.append(now)
        self._total_ms.append(self.current.total_ms)
        self._det_ms.append(self.current.detect_ms)
        self._dep_ms.append(self.current.depth_ms)

        return self.current

    # -----------------------------------------------------------------------
    # Rolling statistics
    # -----------------------------------------------------------------------

    @property
    def fps(self) -> float:
        """Rolling FPS over the last `window` frames."""
        if len(self._frame_ts) < 2:
            return 0.0
        elapsed = self._frame_ts[-1] - self._frame_ts[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_ts) - 1) / elapsed

    @property
    def avg_total_ms(self) -> float:
        """Average end-to-end latency per frame (ms)."""
        return sum(self._total_ms) / len(self._total_ms) if self._total_ms else 0.0

    @property
    def avg_detect_ms(self) -> float:
        """Average YOLO detection latency (ms)."""
        return sum(self._det_ms) / len(self._det_ms) if self._det_ms else 0.0

    @property
    def avg_depth_ms(self) -> float:
        """Average depth inference latency on frames where depth ran (ms)."""
        ran = [v for v in self._dep_ms if v > 0]
        return sum(ran) / len(ran) if ran else 0.0

    @property
    def frame_count(self) -> int:
        """Total frames processed since creation."""
        return self._frame_num

    def summary_line(self) -> str:
        """One-line summary for console display."""
        return (
            f"FPS:{self.fps:5.1f} | "
            f"Det:{self.avg_detect_ms:5.0f}ms | "
            f"Depth:{self.avg_depth_ms:5.0f}ms | "
            f"Total:{self.avg_total_ms:5.0f}ms | "
            f"Frame:{self.frame_count}"
        )

    def __repr__(self) -> str:
        return f"PerfMonitor(fps={self.fps:.1f}, frames={self.frame_count})"
