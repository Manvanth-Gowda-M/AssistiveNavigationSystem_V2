"""
Unit Tests — Phase 14 Performance Optimisations
=================================================
Verifies:
  1. PerfMonitor has CPU/RAM fields (OPT-2)
  2. PerfMonitor writes CSV correctly (OPT-6)
  3. DepthEstimator reads depth.num_threads not detection.num_threads (OPT-1)
  4. depth_update_interval preserved after OPT-1
  5. detection.input_size default is still 640 (OPT-3 — no default change)
  6. depth_update_interval default is still 5 (OPT-4 — no default change)
  7. debug_view skips frame.copy() when disabled (OPT-5)
  8. Functional equivalence — detection/tracking/depth still work

Run:
    .venv/Scripts/python.exe -m pytest tests/unit/test_phase14.py -v
"""

import time
import pytest
import numpy as np
from pathlib import Path


@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


# ===========================================================================
# OPT-1: depth.num_threads separate from detection.num_threads
# ===========================================================================

class TestOPT1DepthThreads:

    def test_depth_config_has_num_threads(self, config):
        """depth.num_threads must exist and default to 0 (all threads)."""
        depth_cfg = config.get("depth", {})
        assert "num_threads" in depth_cfg, (
            "depth.num_threads key missing from config.yaml"
        )
        assert depth_cfg["num_threads"] == 0, (
            f"depth.num_threads should be 0 (all threads), got {depth_cfg['num_threads']}"
        )

    def test_depth_estimator_reads_depth_num_threads(self):
        """DepthEstimator must read depth.num_threads, not detection.num_threads."""
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        # Config with different values for each — verifies correct source
        cfg = {
            "detection": {"num_threads": 99},   # should NOT be used by depth
            "depth": {
                "num_threads": 6,               # should be used
                "model_path": "models/depth_anything_v2_small.onnx",
                "input_size": 252,
                "depth_update_interval": 5,
                "roi_center_fraction": 0.5,
                "min_valid_samples": 10,
                "smoothing_window": 5,
            },
            "proximity": {},
        }
        est = DepthEstimator(cfg)
        assert est._num_threads == 6, (
            f"DepthEstimator should use depth.num_threads=6, "
            f"but got {est._num_threads} (reading detection.num_threads=99 is wrong)"
        )

    def test_depth_estimator_default_all_threads(self):
        """When depth.num_threads is absent, default must be 0 (all threads)."""
        from assistive_navigation.depth.depth_estimator import DepthEstimator
        cfg = {
            "detection": {"num_threads": 4},
            "depth": {
                # num_threads intentionally absent — should default to 0
                "model_path": "models/depth_anything_v2_small.onnx",
                "input_size": 252,
                "depth_update_interval": 5,
                "roi_center_fraction": 0.5,
                "min_valid_samples": 10,
                "smoothing_window": 5,
            },
            "proximity": {},
        }
        est = DepthEstimator(cfg)
        assert est._num_threads == 0, (
            f"Default depth.num_threads should be 0, got {est._num_threads}"
        )

    def test_depth_update_interval_preserved(self, config):
        """depth_update_interval must still be 5 after OPT-1."""
        assert config["depth"]["depth_update_interval"] == 5

    def test_detection_num_threads_unchanged(self, config):
        """detection.num_threads must still be 4 (OPT-1 must not change it)."""
        assert config["detection"]["num_threads"] == 4


# ===========================================================================
# OPT-2 / OPT-6: PerfMonitor CPU/RAM + CSV
# ===========================================================================

class TestOPT2PerfMonitor:

    def test_frame_stats_has_cpu_field(self):
        """FrameStats must have cpu_percent field."""
        from assistive_navigation.utils.perf_monitor import FrameStats
        fs = FrameStats()
        assert hasattr(fs, "cpu_percent")
        assert isinstance(fs.cpu_percent, float)

    def test_frame_stats_has_rss_field(self):
        """FrameStats must have rss_mb field."""
        from assistive_navigation.utils.perf_monitor import FrameStats
        fs = FrameStats()
        assert hasattr(fs, "rss_mb")
        assert isinstance(fs.rss_mb, float)

    def test_psutil_available_property(self):
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        m = PerfMonitor()
        assert isinstance(m.psutil_available, bool)
        # psutil IS installed — must be True
        assert m.psutil_available is True

    def test_cpu_percent_populated_after_record(self):
        """cpu_percent must be populated (>= 0) after record()."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        m = PerfMonitor()
        m.start_frame()
        time.sleep(0.01)
        m.record(detect_ms=50.0)
        m.end_frame()
        # cpu_percent is system-wide and non-negative
        assert m.current.cpu_percent >= 0.0

    def test_rss_populated_after_record(self):
        """rss_mb must be > 0 (process uses some RAM)."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        m = PerfMonitor()
        m.start_frame()
        m.record(detect_ms=50.0)
        m.end_frame()
        assert m.current.rss_mb > 0.0

    def test_summary_line_contains_cpu_label(self):
        """Summary line must label CPU as system-wide (avoid misleading label)."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        m = PerfMonitor()
        m.start_frame()
        m.record()
        m.end_frame()
        line = m.summary_line()
        # Must contain the system-wide label
        assert "(sys)" in line, (
            f"summary_line must label CPU as system-wide '(sys)': {line!r}"
        )

    def test_csv_created_and_has_header(self, tmp_path):
        """CSV file must be created with correct header columns."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        csv_path = tmp_path / "test_perf.csv"
        m = PerfMonitor(log_path=str(csv_path))
        m.start_frame()
        m.record(detect_ms=45.0, depth_ms=120.0, depth_ran=True,
                 detections_raw=3, detections_acc=2, tracks_active=2,
                 confirmed_count=1, alert_issued=False)
        m.end_frame()
        m.close()

        assert csv_path.exists(), "CSV file was not created"
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) >= 2, "CSV must have header + at least 1 data row"
        header = lines[0]
        for col in ["frame_num", "total_ms", "detect_ms", "depth_ms",
                    "cpu_percent_system", "rss_mb_process"]:
            assert col in header, f"CSV header missing column: {col}"

    def test_csv_column_cpu_labeled_system(self, tmp_path):
        """CPU column must be named cpu_percent_system (not cpu_percent_process)."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        csv_path = tmp_path / "cpu_label.csv"
        m = PerfMonitor(log_path=str(csv_path))
        m.start_frame(); m.record(); m.end_frame(); m.close()
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        assert "cpu_percent_system" in header, (
            "CPU column must be 'cpu_percent_system' to indicate it is system-wide"
        )
        assert "cpu_percent_process" not in header, (
            "Must not claim CPU is process-specific"
        )

    def test_csv_data_row_written(self, tmp_path):
        """Each end_frame() call must write one row to CSV."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        csv_path = tmp_path / "rows.csv"
        m = PerfMonitor(log_path=str(csv_path))
        for _ in range(3):
            m.start_frame(); m.record(); m.end_frame()
        m.close()
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 4, f"Expected 1 header + 3 data rows, got {len(lines)}"

    def test_csv_no_crash_without_log_path(self):
        """PerfMonitor with no log_path must work without writing CSV."""
        from assistive_navigation.utils.perf_monitor import PerfMonitor
        m = PerfMonitor()   # no log_path
        m.start_frame(); m.record(detect_ms=50.0); m.end_frame()
        m.close()  # must not raise


# ===========================================================================
# OPT-3: detection.input_size default preserved at 640
# ===========================================================================

class TestOPT3DetectionInputSize:

    def test_detection_input_size_default_640(self, config):
        """detection.input_size must remain 640 (OPT-3: no default change)."""
        assert config["detection"]["input_size"] == 640, (
            f"detection.input_size must be 640, got {config['detection']['input_size']}. "
            "OPT-3 is documentation only — the default must not change."
        )


# ===========================================================================
# OPT-4: depth_update_interval default preserved at 5
# ===========================================================================

class TestOPT4DepthInterval:

    def test_depth_update_interval_default_5(self, config):
        """depth_update_interval must remain 5 (OPT-4: no default change)."""
        assert config["depth"]["depth_update_interval"] == 5

    def test_depth_interval_works_with_loaded_estimator(self, config):
        """
        With OPT-1 (num_threads=0), depth_update_interval must still be
        respected: depth runs every 5th frame.
        """
        from assistive_navigation.depth.depth_estimator import DepthEstimator, DepthEstimatorError
        try:
            est = DepthEstimator(config)
            est.load()
        except DepthEstimatorError as e:
            pytest.skip(f"Depth model not available: {e}")

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_ran = []
        interval  = est.update_interval

        for _ in range(interval * 3):
            prev = est.inference_count
            est.process_frame(blank)
            depth_ran.append(est.inference_count > prev)

        expected = interval * 3 // interval   # = 3 inferences
        actual   = sum(depth_ran)
        assert actual == expected, (
            f"Expected {expected} depth inferences in {interval*3} frames "
            f"(interval={interval}), got {actual}. "
            f"depth_update_interval may have been broken by OPT-1."
        )


# ===========================================================================
# OPT-5: debug_view skips frame.copy() when disabled
# ===========================================================================

class TestOPT5DebugViewCopy:

    def test_draw_returns_none_when_disabled(self, config):
        """draw() must return None when debug is disabled."""
        from assistive_navigation.visualization.debug_view import DebugView
        # Override to disabled
        cfg = dict(config)
        cfg["debug"] = dict(config.get("debug", {}))
        cfg["debug"]["enabled"] = False
        view = DebugView(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = view.draw(frame, [], None, None)
        assert result is None

    def test_draw_returns_none_when_show_window_false(self):
        """draw() must return None when show_window=False and no boxes."""
        from assistive_navigation.visualization.debug_view import DebugView
        cfg = {
            "debug": {
                "enabled": True,
                "show_fps": False,
                "show_boxes": False,
                "show_labels": False,
                "show_track_id": False,
                "show_direction": False,
                "show_proximity": False,
                "show_priority": False,
                "window_title": "Test",
            },
            "pipeline": {"show_debug_window": False},
        }
        view = DebugView(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # With show_boxes=False and show_fps=False, draw returns None
        result = view.draw(frame, [], None, None)
        assert result is None, (
            "draw() should return None when all overlays are disabled "
            "(avoids unnecessary frame.copy())"
        )

    def test_draw_returns_array_when_enabled_with_boxes(self, config):
        """draw() must return an ndarray when boxes are enabled."""
        from assistive_navigation.visualization.debug_view import DebugView
        cfg = dict(config)
        cfg["debug"] = dict(config.get("debug", {}))
        cfg["debug"]["enabled"]     = True
        cfg["debug"]["show_boxes"]  = True
        cfg["debug"]["show_fps"]    = True
        cfg["pipeline"] = {"show_debug_window": True}
        view = DebugView(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = view.draw(frame, [], None, None)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_original_frame_not_modified(self, config):
        """draw() must not modify the original frame."""
        from assistive_navigation.visualization.debug_view import DebugView
        cfg = dict(config)
        view = DebugView(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_sum = frame.sum()
        view.draw(frame, [], None, None)
        assert frame.sum() == original_sum, "draw() must not modify the original frame"


# ===========================================================================
# Functional equivalence — depth still works after OPT-1
# ===========================================================================

class TestFunctionalEquivalence:

    def test_depth_estimator_loads_with_new_config(self, config):
        """DepthEstimator must load successfully with depth.num_threads=0."""
        from assistive_navigation.depth.depth_estimator import DepthEstimator, DepthEstimatorError
        try:
            est = DepthEstimator(config)
            est.load()
        except DepthEstimatorError as e:
            pytest.skip(f"Depth model not available: {e}")
        assert est.is_loaded is True
        assert est._num_threads == 0   # OPT-1 confirmed active

    def test_depth_output_valid_after_opt1(self, config):
        """Depth map output must still be valid float32 0-1 array after OPT-1."""
        from assistive_navigation.depth.depth_estimator import DepthEstimator, DepthEstimatorError
        try:
            est = DepthEstimator(config)
            est.load()
        except DepthEstimatorError as e:
            pytest.skip(f"Depth model not available: {e}")

        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = est.process_frame(frame, force=True)
        assert result is not None
        assert result.ndim == 2
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

    def test_detection_unchanged(self, config):
        """Detection input_size=640 must still produce valid results."""
        from assistive_navigation.detection.detector import ObjectDetector, DetectorError
        try:
            det = ObjectDetector(config)
            det.load()
        except DetectorError as e:
            pytest.skip(f"Detector not available: {e}")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = det.detect(frame)
        assert result is not None
        assert result.frame_width  == 640
        assert result.frame_height == 480

    def test_pipeline_intact_after_all_opts(self, config):
        """Full pipeline with all OPT-1..5 applied must still run without error."""
        from assistive_navigation.main import AssistiveNavigationPipeline, FrameResult
        p = AssistiveNavigationPipeline(config, headless=True)

        # Mock camera + detector for this test (no hardware needed)
        from unittest.mock import MagicMock
        from assistive_navigation.detection.detector import DetectionResult
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        p._cam = MagicMock()
        p._cam.read.return_value = blank
        p._cam.is_open = True
        p._cam.actual_width = 640; p._cam.actual_height = 480
        p._cam.release = MagicMock()
        p._detector = MagicMock()
        p._detector.detect.return_value = DetectionResult(
            detections=[], latency_ms=45.0, frame_width=640, frame_height=480, raw_count=0
        )
        p._detector.is_loaded = True
        p._queue = MagicMock(); p._queue.is_running=True
        p._queue.start=MagicMock(); p._queue.stop=MagicMock(); p._queue.put=MagicMock()
        p._tts = MagicMock(); p._tts.is_available=True; p._tts.shutdown=MagicMock()
        p._alert.set_queue(p._queue)
        p._running = True

        result = p.run_frame()
        assert isinstance(result, FrameResult)
        assert result.camera_ok is True
