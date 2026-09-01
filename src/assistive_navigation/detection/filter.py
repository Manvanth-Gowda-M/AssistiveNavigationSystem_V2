"""
Detection Filter Module
=======================
Applies navigation policy to raw detector output.

Phase: 3
Status: IMPLEMENTED

The three-layer model — why it matters:
    Layer 1 (RAW):       ObjectDetector returns everything above 0.15 conf.
                         This is what YOLO actually detected.
    Layer 2 (CONFIDENT): DetectionFilter applies the confidence_threshold
                         from config (default 0.35). Still includes all
                         classes.
    Layer 3 (FILTERED):  DetectionFilter further applies the navigation
                         class list from config.yaml:
                           - navigation_critical, contextual, low_priority
                             classes are ACCEPTED
                           - ignored classes are REJECTED (with reason logged)
                           - any class not in any list is REJECTED

This separation means false positives are NEVER silently hidden.
Every rejected detection is recorded in FilterResult.rejected with
a reason code. The caller can log, analyse, and investigate these.

What this module does NOT do:
    - Does not run YOLO inference (that is ObjectDetector)
    - Does not apply temporal filtering (Phase 10)
    - Does not track objects (Phase 5)
    - Does not assign direction or depth (Phases 6–8)

Usage:
    from assistive_navigation.detection.filter import DetectionFilter
    from assistive_navigation.detection.detector import ObjectDetector

    config = load_config()
    detector = ObjectDetector(config)
    detector.load()
    det_filter = DetectionFilter(config)

    result = detector.detect(frame)
    filtered = det_filter.filter(result)

    for det in filtered.accepted:
        print(f"ACCEPTED: {det.class_name} (priority: {filtered.get_priority(det)})")
    for r in filtered.rejected:
        print(f"REJECTED: {r['class_name']} conf={r['confidence']:.2f} reason={r['reason']}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from assistive_navigation.detection.detector import Detection, DetectionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority and rejection constants
# ---------------------------------------------------------------------------

class Priority:
    """
    Navigation priority levels for detected objects.

    A = CRITICAL  : always alert regardless of distance (person, chair, car)
    B = CONTEXTUAL: alert when CLOSE or nearer (backpack, bottle)
    C = LOW       : alert only when VERY CLOSE (bench, mouse)
    D = IGNORED   : never alert — suppressed to prevent false positive noise
    UNKNOWN       : class not in any configured list
    """
    CRITICAL    = "navigation_critical"
    CONTEXTUAL  = "contextual"
    LOW         = "low_priority"
    IGNORED     = "ignored"
    UNKNOWN     = "unknown"


class RejectionReason:
    """
    Codes recorded for every rejected detection.
    Used for false-positive analysis and debugging.
    """
    LOW_CONFIDENCE      = "low_confidence"     # Below confidence_threshold
    CLASS_IGNORED       = "class_ignored"      # In the 'ignored' list in config
    CLASS_NOT_IN_LIST   = "class_not_in_list"  # Not in any navigation category


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """
    The complete output of one call to DetectionFilter.filter().

    Fields:
        accepted        : Detections that passed all filters.
                          These are what the navigation system acts on.
        rejected        : ALL rejected detections with reason and metadata.
                          Never empty if there were raw detections — every
                          non-accepted detection appears here.
        raw_count       : Total detections from the detector before filtering.
        accepted_count  : Number of accepted detections.
        rejected_count  : Number of rejected detections.
        priority_map    : Maps Detection -> Priority string for accepted items.
    """
    accepted:        List[Detection]
    rejected:        List[dict]
    raw_count:       int
    accepted_count:  int
    rejected_count:  int
    priority_map:    Dict[int, str]  # id(Detection) -> priority string

    def get_priority(self, det: Detection) -> str:
        """Return the priority level for an accepted detection."""
        return self.priority_map.get(id(det), Priority.UNKNOWN)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"FilterResult: raw={self.raw_count}, "
            f"accepted={self.accepted_count}, "
            f"rejected={self.rejected_count} "
            f"({self._rejection_breakdown()})"
        )

    def _rejection_breakdown(self) -> str:
        """Count rejections by reason."""
        counts: Dict[str, int] = {}
        for r in self.rejected:
            reason = r.get("reason", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return ", ".join(f"{k}={v}" for k, v in counts.items())


# ---------------------------------------------------------------------------
# DetectionFilter class
# ---------------------------------------------------------------------------

class DetectionFilter:
    """
    Filters ObjectDetector output according to the navigation class config.

    The filter applies three sequential checks:
      1. Confidence threshold — must be >= detection.confidence_threshold
      2. Ignored classes — rejected if class is in the 'ignored' list
      3. Navigation list — accepted only if class is in one of the
         navigation categories (A, B, or C)

    All rejections are recorded with a reason code in FilterResult.rejected.
    Nothing is silently discarded.

    The navigation class list is read from config.yaml at construction time.
    Changing config.yaml and creating a new DetectionFilter will apply
    the new list without modifying any code.

    Args:
        config (dict): Full configuration dictionary from config.yaml.
    """

    def __init__(self, config: dict) -> None:
        det_cfg = config.get("detection", {})
        self._conf_threshold: float = float(
            det_cfg.get("confidence_threshold", 0.35)
        )

        # Build lookup tables from the object_classes config section.
        # Each table maps class_id (int) -> class_name (str).
        obj_cfg = config.get("object_classes", {})

        self._critical:    Dict[int, str] = self._parse_class_list(
            obj_cfg.get("navigation_critical", [])
        )
        self._contextual:  Dict[int, str] = self._parse_class_list(
            obj_cfg.get("contextual", [])
        )
        self._low_priority: Dict[int, str] = self._parse_class_list(
            obj_cfg.get("low_priority", [])
        )
        self._ignored:     Dict[int, str] = self._parse_class_list(
            obj_cfg.get("ignored", [])
        )

        # Combined set of all known class IDs (for "not in any list" check)
        self._all_known_ids: Set[int] = (
            set(self._critical.keys())
            | set(self._contextual.keys())
            | set(self._low_priority.keys())
            | set(self._ignored.keys())
        )

        logger.debug(
            "DetectionFilter configured — conf_threshold=%.2f, "
            "critical=%d, contextual=%d, low=%d, ignored=%d classes",
            self._conf_threshold,
            len(self._critical), len(self._contextual),
            len(self._low_priority), len(self._ignored)
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def filter(self, detection_result: DetectionResult) -> FilterResult:
        """
        Apply navigation policy to a DetectionResult.

        Processes each detection through three sequential gates:
          Gate 1: confidence >= threshold
          Gate 2: class not in ignored list
          Gate 3: class is in a navigation category (A, B, or C)

        Every detection that fails any gate is recorded in
        FilterResult.rejected with a specific reason code.

        Args:
            detection_result: Output from ObjectDetector.detect()

        Returns:
            FilterResult with accepted/rejected lists and metadata.
        """
        accepted: List[Detection] = []
        rejected: List[dict] = []
        priority_map: Dict[int, str] = {}

        for det in detection_result.detections:

            # --- Gate 1: Confidence threshold ---
            if det.confidence < self._conf_threshold:
                rejected.append({
                    "class_id":   det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "reason":     RejectionReason.LOW_CONFIDENCE,
                    "threshold":  self._conf_threshold,
                    "bbox":       det.bbox,
                })
                logger.debug(
                    "REJECTED [low_confidence]: %s conf=%.2f < %.2f",
                    det.class_name, det.confidence, self._conf_threshold
                )
                continue

            # --- Gate 2: Ignored class ---
            if det.class_id in self._ignored:
                rejected.append({
                    "class_id":   det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "reason":     RejectionReason.CLASS_IGNORED,
                    "bbox":       det.bbox,
                })
                logger.debug(
                    "REJECTED [class_ignored]: %s conf=%.2f — "
                    "suppressed (known false-positive source)",
                    det.class_name, det.confidence
                )
                continue

            # --- Gate 3: Must be in a navigation category ---
            priority = self._get_priority(det.class_id)

            if priority == Priority.UNKNOWN:
                rejected.append({
                    "class_id":   det.class_id,
                    "class_name": det.class_name,
                    "confidence": det.confidence,
                    "reason":     RejectionReason.CLASS_NOT_IN_LIST,
                    "bbox":       det.bbox,
                })
                logger.debug(
                    "REJECTED [class_not_in_list]: %s conf=%.2f — "
                    "not in any navigation category",
                    det.class_name, det.confidence
                )
                continue

            # --- Passed all gates ---
            accepted.append(det)
            priority_map[id(det)] = priority
            logger.debug(
                "ACCEPTED [%s]: %s conf=%.2f",
                priority, det.class_name, det.confidence
            )

        result = FilterResult(
            accepted=accepted,
            rejected=rejected,
            raw_count=detection_result.raw_count,
            accepted_count=len(accepted),
            rejected_count=len(rejected),
            priority_map=priority_map,
        )

        logger.debug(result.summary())
        return result

    def get_all_accepted_ids(self) -> Set[int]:
        """Return all class IDs that will be accepted (A + B + C, not D)."""
        return (
            set(self._critical.keys())
            | set(self._contextual.keys())
            | set(self._low_priority.keys())
        )

    def get_ignored_ids(self) -> Set[int]:
        """Return all class IDs that are permanently suppressed."""
        return set(self._ignored.keys())

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_class_list(class_list: list) -> Dict[int, str]:
        """
        Convert a YAML class list (list of {id, name} dicts) into
        an {id: name} dictionary.

        Example input:
            [{id: 0, name: "person"}, {id: 56, name: "chair"}]
        Output:
            {0: "person", 56: "chair"}
        """
        result = {}
        for item in class_list:
            if isinstance(item, dict):
                cls_id = item.get("id")
                cls_name = item.get("name", f"class_{cls_id}")
                if cls_id is not None:
                    result[int(cls_id)] = str(cls_name)
        return result

    def _get_priority(self, class_id: int) -> str:
        """
        Return the priority level for a class ID.
        Returns Priority.UNKNOWN if the class is not in any list.
        """
        if class_id in self._critical:
            return Priority.CRITICAL
        if class_id in self._contextual:
            return Priority.CONTEXTUAL
        if class_id in self._low_priority:
            return Priority.LOW
        # Ignored classes have already been filtered out before this is called
        return Priority.UNKNOWN
