"""
Debug Visualization Module
===========================
Draws a debug overlay on camera frames and displays them in an OpenCV window.

Phase: 13
Status: IMPLEMENTED

All overlay elements are individually toggleable via config.yaml debug: section.
In production/audio-only mode, set debug.enabled=false or
pipeline.show_debug_window=false to skip all visualization.

Overlay colour conventions:
  Green   : accepted + confirmed
  Amber   : accepted but candidate (not yet confirmed)
  Grey    : accepted but NEW/LOST state
  Red dim : rejected (below confidence or ignored class)
  Yellow  : HUD text (FPS, alert, stage counts)

What this module does NOT do:
  - Does not run any detection or tracking (those are separate modules)
  - Does not affect the pipeline data flow
  - Does not log to files (that is logger.py / PerfMonitor)

Usage:
    from assistive_navigation.visualization.debug_view import DebugView

    view = DebugView(config)
    # Each frame:
    view.draw(frame, confirmed_objects, alert_result, perf)
    quit = view.show()   # returns True if user pressed Q
    view.close()
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Colours (BGR)
_GREEN  = (0,  200,  0)
_AMBER  = (0,  165, 255)
_GREY   = (120, 120, 120)
_RED    = (60,   60, 200)
_YELLOW = (0,  220, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,    0,   0)


class DebugView:
    """
    OpenCV debug window with pipeline overlay.

    Args:
        config (dict): Full configuration from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        dbg = config.get("debug", {})
        pip = config.get("pipeline", {})

        self._enabled     = bool(dbg.get("enabled",          True))
        self._show_window = bool(pip.get("show_debug_window", True))
        self._title       = str(dbg.get("window_title",       "Assistive Navigation V2 — Debug"))

        self._show_fps      = bool(dbg.get("show_fps",       True))
        self._show_boxes    = bool(dbg.get("show_boxes",     True))
        self._show_labels   = bool(dbg.get("show_labels",    True))
        self._show_track_id = bool(dbg.get("show_track_id",  True))
        self._show_direction= bool(dbg.get("show_direction", True))
        self._show_proximity= bool(dbg.get("show_proximity", True))
        self._show_priority = bool(dbg.get("show_priority",  False))

        self._active = False   # True after first show()
        logger.debug(
            "DebugView created — enabled=%s, show_window=%s",
            self._enabled, self._show_window
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        confirmed_objects: list,    # List[ConfirmedObject]
        alert_result=None,          # AlertResult | None
        perf=None,                  # PerfMonitor | None
    ) -> Optional[np.ndarray]:
        """
        Draw all enabled overlays onto a copy of frame.

        Returns the annotated frame (does NOT modify the original).
        Returns None if the debug view is disabled.

        Args:
            frame:             Raw BGR camera frame.
            confirmed_objects: List[ConfirmedObject] from temporal filter.
            alert_result:      AlertResult from AlertManager (may be None).
            perf:              PerfMonitor for FPS/latency display.
        """
        if not self._enabled:
            return None

        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Draw bounding boxes for each confirmed object
        if self._show_boxes:
            for obj in confirmed_objects:
                self._draw_object_box(overlay, obj)

        # FPS / latency HUD
        if self._show_fps and perf is not None:
            self._draw_hud(overlay, perf, h, w)

        # Alert banner at bottom
        if alert_result is not None and alert_result.alert_issued:
            self._draw_alert_banner(overlay, alert_result.alert_text, h, w)

        # Stage count strip (top-right)
        self._draw_stage_counts(overlay, confirmed_objects, w)

        return overlay

    def show(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Display the frame in the OpenCV window.

        Returns True if the user pressed 'Q' (quit signal).
        Returns False otherwise.

        If frame is None, nothing is shown but the window event loop
        still ticks (required for key detection).
        """
        if not self._enabled or not self._show_window:
            return False

        if not self._active:
            cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
            self._active = True

        if frame is not None:
            cv2.imshow(self._title, frame)

        key = cv2.waitKey(1) & 0xFF
        return key == ord('q') or key == ord('Q')

    def close(self) -> None:
        """Destroy the debug window cleanly."""
        if self._active:
            try:
                cv2.destroyWindow(self._title)
            except Exception:
                pass
            self._active = False
            logger.debug("DebugView: window closed.")

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._show_window

    # -----------------------------------------------------------------------
    # Drawing helpers
    # -----------------------------------------------------------------------

    def _draw_object_box(self, overlay: np.ndarray, obj) -> None:
        """Draw bounding box and label for one ConfirmedObject."""
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        h, w = overlay.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        # Colour by confirmation state
        from assistive_navigation.navigation.temporal import ConfirmationState
        state = getattr(obj, 'confirmation_state', 'NEW')
        if state == ConfirmationState.CONFIRMED:
            colour = _GREEN
            thickness = 2
        elif state == ConfirmationState.CANDIDATE:
            colour = _AMBER
            thickness = 1
        else:
            colour = _GREY
            thickness = 1

        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, thickness)

        # Label line
        parts = []
        if self._show_labels:
            parts.append(f"{obj.class_name} {obj.confidence:.2f}")
        if self._show_track_id:
            parts.append(f"#{obj.track_id}")
        if self._show_direction:
            parts.append(obj.direction)
        if self._show_proximity:
            parts.append(obj.proximity)
        if self._show_priority:
            parts.append(f"s={obj.nav_score:.2f}")

        if parts:
            label = " | ".join(parts)
            label_y = max(y1 - 6, 12)
            cv2.putText(
                overlay, label, (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA
            )

        # Confirmation progress bar (thin bar at bottom of bbox)
        if state == ConfirmationState.CANDIDATE:
            from assistive_navigation.utils.config_loader import load_config
            # Simple ratio: consecutive_count / confirmation_frames
            count = getattr(obj, 'consecutive_count', 0)
            # Use a rough default if we can't get config
            conf_frames = 8
            ratio = min(1.0, count / conf_frames)
            bar_w = int((x2 - x1) * ratio)
            cv2.rectangle(
                overlay,
                (x1, y2 - 3), (x1 + bar_w, y2),
                _AMBER, -1
            )

    def _draw_hud(self, overlay: np.ndarray, perf, h: int, w: int) -> None:
        """Draw FPS and latency info in top-left corner."""
        lines = [
            f"FPS: {perf.fps:.1f}",
            f"Det: {perf.avg_detect_ms:.0f}ms",
        ]
        if perf.avg_depth_ms > 0:
            lines.append(f"Depth: {perf.avg_depth_ms:.0f}ms")
        lines.append(f"Total: {perf.avg_total_ms:.0f}ms")

        for i, line in enumerate(lines):
            cv2.putText(
                overlay, line, (6, 18 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, _YELLOW, 1, cv2.LINE_AA
            )

    def _draw_alert_banner(
        self, overlay: np.ndarray, alert_text: str, h: int, w: int
    ) -> None:
        """Draw the latest alert text as a green banner at the bottom."""
        banner_h = 28
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 80, 0), -1)
        cv2.putText(
            overlay,
            f"ALERT: {alert_text}",
            (8, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _WHITE, 1, cv2.LINE_AA
        )

    def _draw_stage_counts(
        self, overlay: np.ndarray, confirmed_objects: list, w: int
    ) -> None:
        """Draw confirmed/candidate count in top-right."""
        from assistive_navigation.navigation.temporal import ConfirmationState
        n_conf = sum(
            1 for o in confirmed_objects
            if getattr(o, 'confirmation_state', '') == ConfirmationState.CONFIRMED
        )
        n_cand = sum(
            1 for o in confirmed_objects
            if getattr(o, 'confirmation_state', '') == ConfirmationState.CANDIDATE
        )
        text = f"C:{n_conf} P:{n_cand}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.putText(
            overlay, text, (w - tw - 6, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, _YELLOW, 1, cv2.LINE_AA
        )
