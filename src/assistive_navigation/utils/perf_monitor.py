"""
Performance Monitor — Phase 14 Implementation
================================================
Measures per-frame timing, CPU usage, and RAM usage across pipeline stages.
Writes results to logs/performance.csv.

Phase: 14
Status: IMPLEMENTED

CPU usage note (Windows):
    psutil.cpu_percent() reports SYSTEM-WIDE CPU usage on Windows, not
    process-specific CPU. It measures total CPU load across all cores.
    This is the only reliable CPU metric available without kernel-level
    instrumentation on Windows.

RSS memory:
    psutil.Process().memory_info().rss is PROCESS-SPECIFIC resident set
    size — the actual RAM the pipeline process is using.

Phase 14 baseline (i5-12450H, CPU-only, 640×480):
    Detection: ~47ms  |  Depth (when run): ~170ms  |  Total: ~82ms  |  FPS: ~12

Usage:
    from assistive_navigation.utils.perf_monitor import PerfMonitor

    mon = PerfMonitor(log_path="logs/performance.csv")
    mon.start_frame()
    # ... pipeline stages ...
    mon.record(detect_ms=47.0, depth_ms=170.0, depth_ran=True, ...)
    stats = mon.end_frame()
    print(mon.fps, mon.avg_detect_ms)
    mon.close()    # flush and close CSV
"""

import csv
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional


# ---------------------------------------------------------------------------
# Try to import psutil — degrade gracefully if unavailable
# ---------------------------------------------------------------------------
try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
    _PROC = _psutil.Process(os.getpid())
except ImportError:
    _PSUTIL_AVAILABLE = False
    _PROC = None


# ---------------------------------------------------------------------------
# FrameStats — one record per pipeline frame
# ---------------------------------------------------------------------------

@dataclass
class FrameStats:
    """Timing and resource statistics for one pipeline frame."""
    frame_num:        int   = 0
    total_ms:         float = 0.0    # wall-clock: start_frame → end_frame
    detect_ms:        float = 0.0    # YOLO inference latency (ms)
    depth_ms:         float = 0.0    # depth inference latency (0 if skipped)
    depth_ran:        bool  = False  # True if depth inference ran this frame
    detections_raw:   int   = 0
    detections_acc:   int   = 0
    tracks_active:    int   = 0
    confirmed_count:  int   = 0
    alert_issued:     bool  = False
    # Phase 14 additions
    cpu_percent:      float = 0.0    # SYSTEM-WIDE CPU usage % (psutil, Windows)
    rss_mb:           float = 0.0    # Process RSS memory (MB) — process-specific


# ---------------------------------------------------------------------------
# PerfMonitor
# ---------------------------------------------------------------------------

class PerfMonitor:
    """
    Per-frame performance monitor with CSV logging.

    Args:
        window:   Number of recent frames for rolling averages (default 30).
        log_path: Optional path to write performance CSV.
                  Writes a header row on first frame.
    """

    def __init__(
        self,
        window:   int            = 30,
        log_path: Optional[str]  = None,
    ) -> None:
        self._window   = window
        self._log_path = Path(log_path) if log_path else None

        self._frame_ts: Deque[float] = deque(maxlen=window)
        self._total_ms: Deque[float] = deque(maxlen=window)
        self._det_ms:   Deque[float] = deque(maxlen=window)
        self._dep_ms:   Deque[float] = deque(maxlen=window)

        self._frame_num:     int            = 0
        self._t_frame_start: Optional[float] = None

        self.current = FrameStats()

        # CSV writer
        self._csv_file   = None
        self._csv_writer = None
        if self._log_path is not None:
            self._open_csv()

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

        # CPU and RAM (non-blocking, minimal overhead ~0.01ms)
        if _PSUTIL_AVAILABLE:
            # cpu_percent(interval=None) = non-blocking, since-last-call
            self.current.cpu_percent = _psutil.cpu_percent(interval=None)
            if _PROC is not None:
                try:
                    self.current.rss_mb = _PROC.memory_info().rss / 1_048_576
                except Exception:
                    self.current.rss_mb = 0.0

    def end_frame(self) -> FrameStats:
        """
        Mark end of frame, update rolling stats, write CSV row.
        Returns the completed FrameStats.
        """
        if self._t_frame_start is not None:
            self.current.total_ms = (
                time.perf_counter() - self._t_frame_start
            ) * 1000.0

        now = time.perf_counter()
        self._frame_ts.append(now)
        self._total_ms.append(self.current.total_ms)
        self._det_ms.append(self.current.detect_ms)
        self._dep_ms.append(self.current.depth_ms)

        if self._csv_writer is not None:
            self._write_row()

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
        return sum(self._total_ms) / len(self._total_ms) if self._total_ms else 0.0

    @property
    def avg_detect_ms(self) -> float:
        return sum(self._det_ms) / len(self._det_ms) if self._det_ms else 0.0

    @property
    def avg_depth_ms(self) -> float:
        """Average depth inference latency on frames where depth actually ran."""
        ran = [v for v in self._dep_ms if v > 0]
        return sum(ran) / len(ran) if ran else 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_num

    @property
    def psutil_available(self) -> bool:
        """True if psutil is installed and CPU/RAM measurement is active."""
        return _PSUTIL_AVAILABLE

    def summary_line(self) -> str:
        """One-line console summary."""
        cpu_str = f" | CPU:{self.current.cpu_percent:.0f}%(sys)" if _PSUTIL_AVAILABLE else ""
        ram_str = f" RAM:{self.current.rss_mb:.0f}MB" if _PSUTIL_AVAILABLE else ""
        return (
            f"FPS:{self.fps:5.1f} | "
            f"Det:{self.avg_detect_ms:5.0f}ms | "
            f"Depth:{self.avg_depth_ms:5.0f}ms | "
            f"Total:{self.avg_total_ms:5.0f}ms"
            f"{cpu_str}{ram_str} | "
            f"Frame:{self.frame_count}"
        )

    # -----------------------------------------------------------------------
    # CSV logging
    # -----------------------------------------------------------------------

    _CSV_COLUMNS = [
        "frame_num", "total_ms", "detect_ms", "depth_ms", "depth_ran",
        "detections_raw", "detections_acc", "tracks_active",
        "confirmed_count", "alert_issued",
        "cpu_percent_system", "rss_mb_process",
    ]

    def _open_csv(self) -> None:
        """Open the CSV file and write the header."""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file   = open(self._log_path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._CSV_COLUMNS
            )
            self._csv_writer.writeheader()
        except Exception as e:
            # Non-critical — continue without CSV logging
            self._csv_file   = None
            self._csv_writer = None

    def _write_row(self) -> None:
        """Write current frame stats to CSV."""
        try:
            self._csv_writer.writerow({
                "frame_num":           self.current.frame_num,
                "total_ms":            f"{self.current.total_ms:.2f}",
                "detect_ms":           f"{self.current.detect_ms:.2f}",
                "depth_ms":            f"{self.current.depth_ms:.2f}",
                "depth_ran":           self.current.depth_ran,
                "detections_raw":      self.current.detections_raw,
                "detections_acc":      self.current.detections_acc,
                "tracks_active":       self.current.tracks_active,
                "confirmed_count":     self.current.confirmed_count,
                "alert_issued":        self.current.alert_issued,
                "cpu_percent_system":  f"{self.current.cpu_percent:.1f}",
                "rss_mb_process":      f"{self.current.rss_mb:.1f}",
            })
        except Exception:
            pass    # non-critical

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file   = None
            self._csv_writer = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        cpu = f", cpu={self.current.cpu_percent:.0f}%(sys)" if _PSUTIL_AVAILABLE else ""
        return f"PerfMonitor(fps={self.fps:.1f}, frames={self.frame_count}{cpu})"
