"""
High-Performance Pipeline Runner & Diagnostic Studio
=====================================================
Features asynchronous decoupled depth inference, fast YOLO batching,
high-precision visual HUD overlays, and real-time telemetry streaming.
"""

import copy
import logging
import queue
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from assistive_navigation.alerts.alert_manager import AlertManager, AlertResult
from assistive_navigation.camera.capture import CameraCapture
from assistive_navigation.depth.depth_estimator import DepthEstimator
from assistive_navigation.depth.fusion import DepthFusion, FusedObject
from assistive_navigation.detection.detector import Detection, DetectionResult, ObjectDetector
from assistive_navigation.detection.filter import DetectionFilter, FilterResult
from assistive_navigation.navigation.priority import NavigationPriorityEngine, ScoredObject
from assistive_navigation.navigation.spatial import Direction, SpatialReasoner
from assistive_navigation.navigation.temporal import ConfirmedObject, TemporalConfirmationFilter
from assistive_navigation.tracking.tracker import ObjectTracker, TrackedObject
from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.utils.perf_monitor import PerfMonitor
from assistive_navigation.visualization.debug_view import DebugView

logger = logging.getLogger(__name__)


class WebPipelineManager:
    """
    High-performance pipeline manager for the testing web interface.
    Decouples expensive depth inference into an asynchronous background worker
    to achieve 25-30+ FPS on CPU.
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path
        self._config = load_config(config_path) if config_path else load_config()
        self._lock = threading.RLock()

        # Pipeline subcomponents
        self._cam: Optional[CameraCapture] = None
        self._detector: Optional[ObjectDetector] = None
        self._filter: Optional[DetectionFilter] = None
        self._tracker: Optional[ObjectTracker] = None
        self._depth: Optional[DepthEstimator] = None
        self._fusion: Optional[DepthFusion] = None
        self._spatial: Optional[SpatialReasoner] = None
        self._priority: Optional[NavigationPriorityEngine] = None
        self._temporal: Optional[TemporalConfirmationFilter] = None
        self._alert: Optional[AlertManager] = None
        self._debug: Optional[DebugView] = None
        self._perf = PerfMonitor(window=30)

        # Performance profile: 'fast' (480px, 30+ fps), 'balanced' (640px), 'turbo' (320px)
        self._perf_profile: str = "fast"

        # Asynchronous Depth Worker
        self._depth_worker_running = False
        self._depth_worker_thread: Optional[threading.Thread] = None
        self._depth_frame_slot: Optional[np.ndarray] = None
        self._depth_event = threading.Event()
        self._cached_depth_map: Optional[np.ndarray] = None
        self._cached_depth_ms: float = 0.0
        self._cached_depth_ran: bool = False
        self._cached_depth_age: int = 0
        self._depth_lock = threading.Lock()

        # Streaming state
        self._is_streaming = False
        self._stream_thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None
        self._latest_telemetry: Dict[str, Any] = {}
        self._latest_alert: Optional[Dict[str, Any]] = None
        self._source_mode: str = "camera"  # "camera", "mock"
        self._mock_frames: List[np.ndarray] = []
        self._mock_frame_idx: int = 0

        self._init_components()
        self._start_depth_worker()

    def _init_components(self) -> None:
        """Initialise pipeline components."""
        with self._lock:
            # Set input size based on performance profile
            if self._perf_profile == "fast":
                self._config.setdefault("detection", {})["input_size"] = 480
            elif self._perf_profile == "turbo":
                self._config.setdefault("detection", {})["input_size"] = 320

            self._cam = CameraCapture(self._config)
            self._detector = ObjectDetector(self._config)
            self._filter = DetectionFilter(self._config)
            self._tracker = ObjectTracker(self._config)
            self._depth = DepthEstimator(self._config)
            self._fusion = DepthFusion(self._config)
            self._spatial = SpatialReasoner(self._config)
            self._priority = NavigationPriorityEngine(self._config)
            self._temporal = TemporalConfirmationFilter(self._config)
            self._alert = AlertManager(self._config, audio_queue=None)
            self._debug = DebugView(self._config)

            # Load models
            try:
                self._detector.load()
                logger.info("Detector loaded: %s (imgsz=%d)", self._detector.model_name, self._detector.input_size)
            except Exception as e:
                logger.warning("Detector load warning: %s", e)

            try:
                self._depth.load()
                logger.info("Depth estimator loaded (async worker ready).")
            except Exception as e:
                logger.warning("Depth estimator load warning: %s", e)

    def _start_depth_worker(self) -> None:
        """Starts the asynchronous depth inference background worker."""
        if self._depth_worker_running:
            return
        self._depth_worker_running = True
        self._depth_worker_thread = threading.Thread(target=self._depth_worker_loop, daemon=True)
        self._depth_worker_thread.start()
        logger.info("Async Depth Inference Worker started.")

    def _depth_worker_loop(self) -> None:
        """Background thread that computes depth maps asynchronously."""
        while self._depth_worker_running:
            # Wait for a new frame to process
            self._depth_event.wait(timeout=0.2)
            self._depth_event.clear()

            frame_to_process = None
            with self._depth_lock:
                if self._depth_frame_slot is not None:
                    frame_to_process = self._depth_frame_slot
                    self._depth_frame_slot = None

            if frame_to_process is None or self._depth is None or not self._depth.is_loaded:
                continue

            try:
                t0 = time.perf_counter()
                d_map = self._depth.process_frame(frame_to_process)
                latency = (time.perf_counter() - t0) * 1000.0

                with self._depth_lock:
                    self._cached_depth_map = d_map
                    self._cached_depth_ms = round(latency, 1)
                    self._cached_depth_ran = True
                    self._cached_depth_age = 0
            except Exception as e:
                logger.debug("Async depth worker error: %s", e)

    def _submit_frame_to_depth_worker(self, frame: np.ndarray) -> None:
        """Submits frame to async depth worker without blocking."""
        with self._depth_lock:
            self._depth_frame_slot = frame
            self._cached_depth_age += 1
        self._depth_event.set()

    def _get_latest_depth_map(self) -> Tuple[Optional[np.ndarray], float, bool, int]:
        """Fetches latest depth map atomically with zero latency impact."""
        with self._depth_lock:
            return self._cached_depth_map, self._cached_depth_ms, self._cached_depth_ran, self._cached_depth_age

    @property
    def config(self) -> Dict[str, Any]:
        with self._lock:
            cfg = copy.deepcopy(self._config)
            cfg["perf_profile"] = self._perf_profile
            return cfg

    def set_performance_profile(self, profile: str) -> Dict[str, Any]:
        """Switches performance profile ('fast', 'balanced', 'turbo')."""
        with self._lock:
            self._perf_profile = profile
            if profile == "fast":
                self._config.setdefault("detection", {})["input_size"] = 480
            elif profile == "balanced":
                self._config.setdefault("detection", {})["input_size"] = 640
            elif profile == "turbo":
                self._config.setdefault("detection", {})["input_size"] = 320

            if self._detector:
                self._detector._input_size = self._config["detection"]["input_size"]
            return self.config

    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update runtime configuration parameters."""
        with self._lock:
            self._config.update(new_config)
            if "perf_profile" in new_config:
                self.set_performance_profile(new_config["perf_profile"])

            self._filter = DetectionFilter(self._config)
            self._spatial = SpatialReasoner(self._config)
            self._priority = NavigationPriorityEngine(self._config)
            self._temporal = TemporalConfirmationFilter(self._config)
            self._alert = AlertManager(self._config, audio_queue=None)
            self._debug = DebugView(self._config)
            return copy.deepcopy(self._config)

    def start_camera(self) -> bool:
        """Open camera hardware safely."""
        with self._lock:
            if self._cam and not self._cam.is_open:
                try:
                    self._cam.open()
                    return True
                except Exception as e:
                    logger.warning("Camera open failed: %s", e)
                    return False
            return True

    def stop_camera(self) -> None:
        """Release camera hardware."""
        with self._lock:
            if self._cam and self._cam.is_open:
                try:
                    self._cam.release()
                except Exception as e:
                    logger.debug("Camera release error: %s", e)

    def start_streaming(self) -> None:
        """Start background capture and inference loop."""
        with self._lock:
            if self._is_streaming:
                return
            self._is_streaming = True
            if self._source_mode == "camera":
                self.start_camera()
            self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._stream_thread.start()
            logger.info("Pipeline streaming started.")

    def stop_streaming(self) -> None:
        """Stop background capture and inference loop."""
        self._is_streaming = False
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)
        self.stop_camera()
        logger.info("Pipeline streaming stopped.")

    def set_source_mode(self, mode: str, mock_frames: Optional[List[np.ndarray]] = None) -> None:
        """Set source mode ('camera', 'mock')."""
        with self._lock:
            self._source_mode = mode
            if mock_frames:
                self._mock_frames = mock_frames
                self._mock_frame_idx = 0
            if mode != "camera":
                self.stop_camera()
            elif self._is_streaming:
                self.start_camera()

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Retrieve most recent JPEG frame."""
        return self._latest_jpeg

    def get_latest_telemetry(self) -> Dict[str, Any]:
        """Retrieve most recent frame telemetry."""
        with self._lock:
            return copy.deepcopy(self._latest_telemetry)

    def _generate_synthetic_fallback_frame(self) -> np.ndarray:
        """Generates dynamic high-resolution simulation scene when camera is inactive."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Gradient background
        for y in range(480):
            val = int(22 + 18 * (y / 480))
            frame[y, :] = (val, val + 4, val + 12)

        # Tech grid
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (38, 42, 54), 1)
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (38, 42, 54), 1)

        # Moving obstacle
        t = time.time()
        cx = int(320 + 150 * np.sin(t * 1.4))
        cy = int(240 + 40 * np.cos(t * 1.1))
        w, h = 110, 180
        x1, y1 = max(10, cx - w // 2), max(10, cy - h // 2)
        x2, y2 = min(630, cx + w // 2), min(470, cy + h // 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 180, 0), 2)
        cv2.putText(frame, "SIMULATED OBSTACLE (person)", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, "SIMULATION / MOCK CAMERA FEED", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Physical camera disconnected or running in mock mode", (20, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 170, 190), 1, cv2.LINE_AA)
        return frame

    def _stream_loop(self) -> None:
        """Main high-speed capture and inference loop."""
        while self._is_streaming:
            t_start = time.perf_counter()

            frame = None
            if self._source_mode == "camera":
                if self._cam and self._cam.is_open:
                    try:
                        frame = self._cam.read()
                    except Exception as e:
                        logger.debug("Camera read exception: %s", e)
                        frame = None
                if frame is None or np.mean(frame) < 1.0:
                    frame = self._generate_synthetic_fallback_frame()
            elif self._source_mode == "mock" and self._mock_frames:
                frame = self._mock_frames[self._mock_frame_idx % len(self._mock_frames)].copy()
                self._mock_frame_idx += 1
            else:
                frame = self._generate_synthetic_fallback_frame()

            if frame is None:
                time.sleep(0.02)
                continue

            # Submit frame to background depth worker
            self._submit_frame_to_depth_worker(frame)

            # Process frame synchronously through YOLO + tracking + spatial + alert pipeline
            telemetry, annotated_frame = self.process_single_frame(frame)

            # High-speed JPEG encoding with optimal compression quality
            ok, buf = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                self._latest_jpeg = buf.tobytes()

            with self._lock:
                self._latest_telemetry = telemetry
                if telemetry.get("alert_issued"):
                    self._latest_alert = {
                        "text": telemetry.get("alert_text"),
                        "reason": telemetry.get("alert_reason"),
                        "timestamp": time.time(),
                    }

            # Regulate frame rate dynamically based on performance profile
            target_fps = 45.0 if self._perf_profile == "turbo" else (35.0 if self._perf_profile == "fast" else 28.0)
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.001, (1.0 / target_fps) - elapsed)
            time.sleep(sleep_time)

    def process_single_frame(self, frame: np.ndarray) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Executes one full frame across all 11 stages and produces precision telemetry & visual HUD.
        """
        self._perf.start_frame()
        t0 = time.perf_counter()
        fh, fw = frame.shape[:2]

        # 1. Detection
        det_result = None
        detect_ms = 0.0
        raw_detections: List[Dict[str, Any]] = []
        if self._detector and self._detector.is_loaded:
            try:
                det_result = self._detector.detect(frame)
                detect_ms = det_result.latency_ms
                for d in det_result.detections:
                    raw_detections.append({
                        "class_id": d.class_id,
                        "class_name": d.class_name,
                        "confidence": float(d.confidence),
                        "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                        "area_norm": float(d.area_norm),
                        "center_x": float(d.center_x),
                        "center_y": float(d.center_y),
                    })
            except Exception as e:
                logger.debug("Detector error: %s", e)

        # 2. Filter
        filtered_result = self._filter.filter(det_result) if (self._filter and det_result) else None
        accepted = filtered_result.accepted if filtered_result else []
        rejected = filtered_result.rejected if filtered_result else []

        # 3. Tracking
        tracked: List[TrackedObject] = []
        if self._tracker:
            try:
                tracked = self._tracker.update(accepted, fw, fh)
            except Exception as e:
                logger.debug("Tracker error: %s", e)

        # 4. Asynchronous Depth Fetch (0.0ms main-thread cost)
        depth_map, depth_ms, depth_ran, depth_age = self._get_latest_depth_map()

        # 5. Fusion
        fused: List[FusedObject] = []
        if self._fusion:
            try:
                fused = self._fusion.fuse(tracked, depth_map, depth_age)
            except Exception as e:
                logger.debug("Fusion error: %s", e)

        # 6. Spatial
        directed = []
        if self._spatial:
            try:
                directed = self._spatial.assign_direction(fused)
            except Exception as e:
                logger.debug("Spatial error: %s", e)

        # 7. Priority
        scored: List[ScoredObject] = []
        if self._priority:
            try:
                scored = self._priority.score(directed)
            except Exception as e:
                logger.debug("Priority error: %s", e)

        # 8. Temporal Confirmation
        confirmed: List[ConfirmedObject] = []
        if self._temporal:
            try:
                confirmed = self._temporal.update(scored)
            except Exception as e:
                logger.debug("Temporal error: %s", e)

        # 9. Alert Generation
        alert_result: Optional[AlertResult] = None
        if self._alert:
            try:
                alert_result = self._alert.process(confirmed)
            except Exception as e:
                logger.debug("Alert error: %s", e)

        # 10. High-Precision Visual Overlay Rendering
        annotated = frame.copy()
        x_left = int(fw * self._config.get("spatial", {}).get("left_boundary", 0.33))
        x_right = int(fw * self._config.get("spatial", {}).get("right_boundary", 0.67))

        # Precision Sector Guidelines with subtle dashed appearance
        cv2.line(annotated, (x_left, 0), (x_left, fh), (0, 180, 240), 1, cv2.LINE_AA)
        cv2.line(annotated, (x_right, 0), (x_right, fh), (0, 180, 240), 1, cv2.LINE_AA)

        # Sector Labels
        cv2.putText(annotated, "LEFT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 170), 1, cv2.LINE_AA)
        cv2.putText(annotated, "AHEAD (CLEARANCE)", (x_left + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, "RIGHT", (x_right + 15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 170), 1, cv2.LINE_AA)

        # Draw Candidate Tracked Objects (Subtle Thin Box)
        confirmed_ids = {c.track_id for c in confirmed}
        for trk in tracked:
            if trk.track_id not in confirmed_ids:
                tx1, ty1, tx2, ty2 = [int(v) for v in trk.bbox]
                cv2.rectangle(annotated, (tx1, ty1), (tx2, ty2), (180, 180, 180), 1)
                cand_label = f"#{trk.track_id} {trk.class_name} ({int(trk.confidence*100)}%)"
                cv2.putText(annotated, cand_label, (tx1, max(12, ty1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw Confirmed Obstacles with Precision Brackets & Color Codes
        for c in confirmed:
            if hasattr(c, "bbox") and c.bbox:
                x1, y1, x2, y2 = [int(v) for v in c.bbox]
            else:
                x1, y1, x2, y2 = int(getattr(c, "x1", 0)), int(getattr(c, "y1", 0)), int(getattr(c, "x2", 0)), int(getattr(c, "y2", 0))

            prox_val = getattr(c, "proximity", getattr(c, "proximity_band", "MEDIUM"))
            prox_name = prox_val.name if hasattr(prox_val, "name") else str(prox_val)

            dir_val = getattr(c, "direction", "AHEAD")
            dir_name = dir_val.name if hasattr(dir_val, "name") else str(dir_val)

            # Color by proximity
            if prox_name == "VERY_CLOSE":
                color = (40, 40, 255)     # Bright Red
            elif prox_name == "CLOSE":
                color = (0, 165, 255)     # Amber
            elif prox_name == "MEDIUM":
                color = (0, 230, 255)     # Cyan
            else:
                color = (0, 230, 120)     # Emerald Green

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Corner brackets
            d = 12
            cv2.line(annotated, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2 - d, y1), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y1), (x2, y1 + d), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1 + d, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x1, y2), (x1, y2 - d), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
            cv2.line(annotated, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)

            # Center target marker
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 3, (255, 255, 255), -1)

            # Tag label
            frames_str = f"{getattr(c, 'consecutive_count', getattr(c, 'frames_confirmed', 0))}f"
            tag = f"#{c.track_id} {c.class_name} [{prox_name}] {dir_name} ({frames_str})"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), (15, 20, 30), -1)
            cv2.putText(annotated, tag, (x1 + 3, max(12, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Alert banner on video if active
        if alert_result and alert_result.alert_issued and alert_result.alert_text:
            alert_msg = f"VOICE: {alert_result.alert_text.upper()}"
            (aw, ah), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ax1 = (fw - aw) // 2
            cv2.rectangle(annotated, (ax1 - 10, fh - 45), (ax1 + aw + 10, fh - 12), (10, 15, 30), -1)
            cv2.rectangle(annotated, (ax1 - 10, fh - 45), (ax1 + aw + 10, fh - 12), (0, 165, 255), 2)
            cv2.putText(annotated, alert_msg, (ax1, fh - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA)

        total_ms = (time.perf_counter() - t0) * 1000.0
        self._perf.record(
            detect_ms=detect_ms,
            depth_ms=depth_ms,
            depth_ran=depth_ran,
            detections_raw=len(raw_detections),
            detections_acc=len(accepted),
            tracks_active=len(tracked),
            confirmed_count=len(confirmed),
            alert_issued=bool(alert_result and alert_result.alert_issued),
        )
        stats = self._perf.end_frame()

        telemetry = {
            "frame_num": stats.frame_num,
            "fps": round(self._perf.fps, 1),
            "detect_ms": round(detect_ms, 1),
            "depth_ms": round(depth_ms, 1),
            "depth_ran": depth_ran,
            "depth_age": depth_age,
            "total_ms": round(total_ms, 1),
            "perf_profile": self._perf_profile,
            "cpu_percent": round(stats.cpu_percent, 1),
            "ram_mb": round(stats.rss_mb, 1),
            "detections_raw_count": len(raw_detections),
            "detections_accepted_count": len(accepted),
            "detections_rejected_count": len(rejected),
            "tracks_active_count": len(tracked),
            "confirmed_count": len(confirmed),
            "alert_issued": bool(alert_result and alert_result.alert_issued),
            "alert_text": alert_result.alert_text if alert_result else "",
            "alert_reason": alert_result.trigger_reason if alert_result else "",
            "confirmed_objects": [
                {
                    "track_id": c.track_id,
                    "class_name": c.class_name,
                    "confidence": round(float(c.confidence), 2),
                    "direction": c.direction.name if hasattr(c.direction, "name") else str(c.direction),
                    "proximity": c.proximity.name if hasattr(c.proximity, "name") else str(getattr(c, "proximity_band", c.proximity)),
                    "priority_score": round(float(getattr(c, "nav_score", getattr(c, "priority_score", 0.0))), 2),
                    "state": getattr(c, "confirmation_state", getattr(c, "state", "CONFIRMED")),
                    "frames_confirmed": getattr(c, "consecutive_count", getattr(c, "frames_confirmed", 0)),
                    "area_norm": round(float(c.area_norm), 3),
                    "bbox": [int(v) for v in c.bbox] if hasattr(c, "bbox") else [0, 0, 0, 0],
                }
                for c in confirmed
            ],
            "rejected_objects": [
                {
                    "class_name": r.get("class_name", "") if isinstance(r, dict) else getattr(r, "class_name", ""),
                    "reason": r.get("reason", "") if isinstance(r, dict) else getattr(r, "reason", ""),
                    "area_norm": round(float(r.get("area_norm", 0.0) if isinstance(r, dict) else getattr(r, "area_norm", 0.0)), 3),
                    "confidence": round(float(r.get("confidence", 0.0) if isinstance(r, dict) else getattr(r, "confidence", 0.0)), 2),
                }
                for r in rejected
            ],
        }

        return telemetry, annotated

    def inspect_full_stages(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Deep diagnostic visualizer: processes frame and renders visual output
        for EVERY single stage (1 to 11) for step-by-step UI inspection.
        """
        fh, fw = frame.shape[:2]
        stages_output: Dict[str, Any] = {}

        def to_b64(img: np.ndarray) -> str:
            import base64
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"
            return ""

        # Stage 1: Input Frame
        stages_output["stage_1_input"] = {
            "name": "1. Frame Capture",
            "description": f"Input frame resolution: {fw}×{fh}",
            "image": to_b64(frame),
            "data": {"width": fw, "height": fh},
        }

        # Stage 2: Raw Detection
        s2_img = frame.copy()
        det_result = self._detector.detect(frame) if (self._detector and self._detector.is_loaded) else None
        raw_list = []
        if det_result:
            for d in det_result.detections:
                raw_list.append({
                    "class_name": d.class_name,
                    "confidence": round(float(d.confidence), 3),
                    "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                    "area_norm": round(float(d.area_norm), 4),
                })
                cv2.rectangle(s2_img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 255), 2)
                cv2.putText(s2_img, f"{d.class_name} {d.confidence:.2f}",
                            (int(d.x1), max(15, int(d.y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        stages_output["stage_2_detection"] = {
            "name": "2. Object Detection (YOLO11n)",
            "description": f"Extracted {len(raw_list)} raw detections in {det_result.latency_ms if det_result else 0:.1f}ms",
            "image": to_b64(s2_img),
            "data": {"count": len(raw_list), "detections": raw_list},
        }

        # Stage 3: Detection Filter
        s3_img = frame.copy()
        filtered = self._filter.filter(det_result) if (self._filter and det_result) else None
        acc_list, rej_list = [], []
        if filtered:
            for d in filtered.accepted:
                acc_list.append({"class_name": d.class_name, "conf": round(float(d.confidence), 2), "area": round(float(d.area_norm), 3)})
                cv2.rectangle(s3_img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 255, 0), 2)
                cv2.putText(s3_img, f"ACCEPTED: {d.class_name}", (int(d.x1), max(15, int(d.y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            for r in filtered.rejected:
                cls_name = r.get("class_name", "") if isinstance(r, dict) else getattr(r, "class_name", "")
                reason_str = r.get("reason", "") if isinstance(r, dict) else getattr(r, "reason", "")
                area_val = r.get("area_norm", 0.0) if isinstance(r, dict) else getattr(r, "area_norm", 0.0)
                rej_list.append({"class_name": cls_name, "reason": reason_str, "area": round(float(area_val), 3)})
                cv2.rectangle(s3_img, (int(r.get("x1", 0)), int(r.get("y1", 0))), (int(r.get("x2", 0)), int(r.get("y2", 0))), (60, 60, 220), 1)

        stages_output["stage_3_filter"] = {
            "name": "3. Detection Filtering & Area Gate",
            "description": f"Accepted: {len(acc_list)} | Rejected (min_area_norm / class): {len(rej_list)}",
            "image": to_b64(s3_img),
            "data": {"accepted": acc_list, "rejected": rej_list},
        }

        # Stage 4: Tracking (ByteTrack)
        s4_img = frame.copy()
        tracked = self._tracker.update(filtered.accepted, fw, fh) if (self._tracker and filtered) else []
        track_list = []
        for t in tracked:
            track_list.append({"track_id": t.track_id, "class_name": t.class_name, "hits": t.hits})
            cv2.rectangle(s4_img, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), (255, 200, 0), 2)
            cv2.putText(s4_img, f"ID #{t.track_id} {t.class_name}", (int(t.x1), max(15, int(t.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2, cv2.LINE_AA)

        stages_output["stage_4_tracking"] = {
            "name": "4. ByteTrack Multi-Object Tracker",
            "description": f"Active tracks: {len(track_list)}",
            "image": to_b64(s4_img),
            "data": {"tracks": track_list},
        }

        # Stage 5: Depth Estimation (Depth Anything V2)
        depth_map = self._depth.process_frame(frame) if (self._depth and self._depth.is_loaded) else None
        s5_img = np.zeros_like(frame)
        if depth_map is not None:
            d_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            s5_img = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
            for t in tracked:
                cv2.rectangle(s5_img, (int(t.x1), int(t.y1)), (int(t.x2), int(t.y2)), (255, 255, 255), 1)

        stages_output["stage_5_depth"] = {
            "name": "5. Depth Anything V2 (Relative Depth Map)",
            "description": "Monocular relative depth map (warmer colors = closer)",
            "image": to_b64(s5_img),
            "data": {"depth_available": depth_map is not None},
        }

        # Stage 6: Fusion
        fused = self._fusion.fuse(tracked, depth_map, 0) if self._fusion else []
        s6_img = frame.copy()
        fused_list = []
        for f in fused:
            fused_list.append({
                "track_id": f.track_id,
                "class_name": f.class_name,
                "relative_depth": round(float(f.relative_depth), 3) if f.relative_depth is not None else None,
                "proximity_band": f.proximity_band.name if hasattr(f.proximity_band, "name") else str(f.proximity_band),
            })
            cv2.rectangle(s6_img, (int(f.x1), int(f.y1)), (int(f.x2), int(f.y2)), (0, 255, 255), 2)
            cv2.putText(s6_img, f"{f.class_name} [{f.proximity_band.name if hasattr(f.proximity_band, 'name') else f.proximity_band}]",
                        (int(f.x1), max(15, int(f.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        stages_output["stage_6_fusion"] = {
            "name": "6. Depth & Proximity Fusion",
            "description": f"Fused {len(fused_list)} objects with depth sampling",
            "image": to_b64(s6_img),
            "data": {"fused": fused_list},
        }

        # Stage 7: Spatial Reasoning
        directed = self._spatial.assign_direction(fused) if self._spatial else []
        s7_img = frame.copy()
        x_left = int(fw * 0.33)
        x_right = int(fw * 0.67)
        cv2.line(s7_img, (x_left, 0), (x_left, fh), (200, 100, 0), 2)
        cv2.line(s7_img, (x_right, 0), (x_right, fh), (200, 100, 0), 2)
        cv2.putText(s7_img, "LEFT SECTOR", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)
        cv2.putText(s7_img, "CENTER SECTOR (AHEAD)", (x_left + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)
        cv2.putText(s7_img, "RIGHT SECTOR", (x_right + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 0), 2)

        dir_list = []
        for d in directed:
            dir_list.append({"track_id": d.track_id, "direction": d.direction.name, "center_x": round(float(d.center_x), 3)})
            cv2.rectangle(s7_img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), (0, 200, 200), 2)
            cv2.putText(s7_img, f"{d.class_name} -> {d.direction.name}", (int(d.x1), max(15, int(d.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)

        stages_output["stage_7_spatial"] = {
            "name": "7. Spatial Sector Classification",
            "description": "Allocated objects into LEFT / CENTER / RIGHT zones",
            "image": to_b64(s7_img),
            "data": {"directed": dir_list},
        }

        # Stage 8: Priority Engine
        scored = self._priority.score(directed) if self._priority else []
        stages_output["stage_8_priority"] = {
            "name": "8. Navigation Priority Scoring",
            "description": f"Ranked {len(scored)} candidates by danger & proximity tier",
            "image": to_b64(s7_img),
            "data": {"scored": [{"track_id": s.track_id, "score": round(float(s.priority_score), 2), "class": s.class_name} for s in scored]},
        }

        # Stage 9: Temporal Confirmation
        confirmed = self._temporal.update(scored) if self._temporal else []
        s9_img = frame.copy()
        conf_list = []
        for c in confirmed:
            conf_list.append({
                "track_id": c.track_id,
                "class_name": c.class_name,
                "state": c.state.name if hasattr(c.state, "name") else str(c.state),
                "frames_confirmed": c.frames_confirmed,
            })
            cv2.rectangle(s9_img, (int(c.x1), int(c.y1)), (int(c.x2), int(c.y2)), (0, 255, 0), 2)
            cv2.putText(s9_img, f"CONFIRMED ({c.frames_confirmed} frames)", (int(c.x1), max(15, int(c.y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        stages_output["stage_9_temporal"] = {
            "name": "9. Temporal Confirmation Filter",
            "description": f"Verified {len(confirmed)} confirmed persistent objects",
            "image": to_b64(s9_img),
            "data": {"confirmed": conf_list},
        }

        # Stage 10: Alert Generation
        alert_res = self._alert.process(confirmed) if self._alert else None
        stages_output["stage_10_alert"] = {
            "name": "10. Natural Language Alert Manager",
            "description": f"Alert Text: '{alert_res.alert_text if alert_res else ''}'",
            "image": to_b64(s9_img),
            "data": {
                "alert_issued": bool(alert_res and alert_res.alert_issued),
                "alert_text": alert_res.alert_text if alert_res else "",
                "trigger_reason": alert_res.trigger_reason if alert_res else "",
            },
        }

        return stages_output
