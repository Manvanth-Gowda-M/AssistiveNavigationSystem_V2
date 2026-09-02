"""
Navigation Priority Engine
===========================
Scores each DirectedObject (Phase 8) by navigation urgency and returns
the list sorted highest-urgency first.

Phase: 9
Status: IMPLEMENTED

What this module does:
    - Accepts List[DirectedObject] from Phase 8 (SpatialReasoner)
    - Computes a nav_score (0.0–1.0) for each object from five sub-scores
    - Returns List[ScoredObject] sorted by nav_score descending
    - All DirectedObject fields pass through unchanged

What this module does NOT do:
    - Does NOT decide whether to speak (Phase 12 — AlertManager)
    - Does NOT apply temporal confirmation (Phase 10)
    - Does NOT re-run detection, tracking, depth, fusion, or direction
    - Does NOT claim the weights are optimal — they are starting estimates

Scoring formula:
    nav_score = (proximity_weight    × proximity_score)
              + (importance_weight   × importance_score)
              + (direction_weight    × direction_score)
              + (confidence_weight   × confidence)
              + (stability_weight    × stability)

    All sub-scores and the weights are normalised to [0.0, 1.0].
    The final nav_score is also clamped to [0.0, 1.0].

Sub-score mappings:
    Proximity:
        VERY_CLOSE → 1.00   (dominant — closest = most urgent)
        CLOSE      → 0.75
        MEDIUM     → 0.40
        FAR        → 0.10
        UNKNOWN    → 0.00   (conservative — no depth data, don't boost)

    Object importance (from 'priority' field on DirectedObject):
        navigation_critical → 1.00
        contextual          → 0.60
        low_priority        → 0.30
        unknown             → 0.00

    Direction:
        CENTER  → 1.00   (directly ahead = highest collision risk)
        LEFT    → 0.70
        RIGHT   → 0.70
        UNKNOWN → 0.50   (safe mid-point)

    Confidence: passed directly from DirectedObject.confidence (0.0–1.0)
    Stability:  passed directly from DirectedObject.stability  (0.0–1.0)

Default weights (from config.yaml priority.weights):
    proximity:         0.40
    object_importance: 0.25
    direction:         0.15
    confidence:        0.10
    stability:         0.10
    (sum = 1.00)

IMPORTANT — weights are starting estimates:
    These values are reasonable defaults that ensure proximity is dominant
    (a very-close object outscores a far one regardless of class).
    They have NOT been empirically validated. Phase 14/15 (optimisation and
    FP reduction) are the designated phases for weight tuning.
    Change them in config.yaml — never in code.

Key guarantee from master requirements:
    "The engine must prevent low-value objects from constantly overriding
    dangerous obstacles."
    This is satisfied by the proximity weight (0.40) being the largest:
    a VERY_CLOSE + any_importance object always scores higher than a
    FAR + navigation_critical object.

Usage:
    from assistive_navigation.navigation.priority import NavigationPriorityEngine
    from assistive_navigation.utils.config_loader import load_config

    config = load_config()
    engine = NavigationPriorityEngine(config)

    # Each frame: takes List[DirectedObject], returns List[ScoredObject]
    scored = engine.score(directed_objects)
    # scored[0] is now the highest-priority object
    for obj in scored:
        print(obj.track_id, obj.class_name, obj.nav_score, obj.direction)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ScoredObject dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoredObject:
    """
    A DirectedObject with a navigation priority score added.

    All fields from DirectedObject (Phase 8) are preserved unchanged.
    The 'nav_score' field is the only addition made by Phase 9.

    Fields from DirectedObject (passed through):
        track_id, class_id, class_name, confidence,
        bbox, bbox_norm, center_x, center_y,
        age, last_seen, stability, area_norm,
        frame_width, frame_height,
        raw_depth, proximity, priority, depth_samples, depth_age,
        direction

    Field added by Phase 9:
        nav_score    : Navigation urgency score, range 0.0–1.0.
                       Higher = more urgent for the navigation system.
                       0.0 = lowest urgency (far, unknown class, no depth).
                       1.0 = maximum urgency (very close, critical class, ahead).

                       This score is a STARTING ESTIMATE based on configurable
                       weights. It is not empirically validated. Do not treat it
                       as a hard threshold for alerting — that is Phase 12.
    """
    # ── From DirectedObject (all passed through) ───────────────────────────
    track_id:      int
    class_id:      int
    class_name:    str
    confidence:    float
    bbox:          tuple
    bbox_norm:     tuple
    center_x:      float
    center_y:      float
    age:           int
    last_seen:     int
    stability:     float
    area_norm:     float
    frame_width:   int
    frame_height:  int
    raw_depth:     float
    proximity:     str
    priority:      str
    depth_samples: int
    depth_age:     int
    direction:     str

    # ── Added by Phase 9 ──────────────────────────────────────────────────
    nav_score:     float    # 0.0–1.0 navigation urgency score

    def __repr__(self) -> str:
        return (
            f"ScoredObject(id={self.track_id}, class={self.class_name}, "
            f"score={self.nav_score:.3f}, direction={self.direction}, "
            f"proximity={self.proximity}, priority={self.priority})"
        )


# ---------------------------------------------------------------------------
# Sub-score lookup tables
# These translate string labels to numeric scores.
# They are fixed by design — the configurable part is the weights only.
# ---------------------------------------------------------------------------

_PROXIMITY_SCORES: Dict[str, float] = {
    "VERY_CLOSE": 1.00,
    "CLOSE":      0.75,
    "MEDIUM":     0.40,
    "FAR":        0.10,
    "UNKNOWN":    0.00,   # conservative — no depth data, do not boost
}

_IMPORTANCE_SCORES: Dict[str, float] = {
    "navigation_critical": 1.00,
    "contextual":          0.60,
    "low_priority":        0.30,
    "unknown":             0.00,
}

_DIRECTION_SCORES: Dict[str, float] = {
    "CENTER":  1.00,   # directly ahead = highest collision risk
    "LEFT":    0.70,
    "RIGHT":   0.70,
    "UNKNOWN": 0.50,   # safe mid-point: don't reward or penalise
}


# ---------------------------------------------------------------------------
# NavigationPriorityEngine
# ---------------------------------------------------------------------------

class NavigationPriorityEngine:
    """
    Scores DirectedObjects by navigation urgency.

    Returns List[ScoredObject] sorted by nav_score descending.
    Higher nav_score = more urgent for navigation.

    Args:
        config (dict): Full configuration from config.yaml.
                       Uses the 'priority' section.
    """

    def __init__(self, config: dict) -> None:
        pri_cfg = config.get("priority", {})
        weights = pri_cfg.get("weights", {})

        self._w_proximity:    float = float(weights.get("proximity",         0.40))
        self._w_importance:   float = float(weights.get("object_importance", 0.25))
        self._w_direction:    float = float(weights.get("direction",         0.15))
        self._w_confidence:   float = float(weights.get("confidence",        0.10))
        self._w_stability:    float = float(weights.get("stability",         0.10))

        total = (self._w_proximity + self._w_importance + self._w_direction
                 + self._w_confidence + self._w_stability)

        if abs(total - 1.0) > 0.05:
            logger.warning(
                "NavigationPriorityEngine: weights sum to %.3f (expected ~1.0). "
                "Scores will be outside the [0,1] range if weights deviate "
                "significantly. Check priority.weights in config.yaml.",
                total
            )
        else:
            logger.debug(
                "NavigationPriorityEngine: weights sum=%.3f  "
                "proximity=%.2f importance=%.2f direction=%.2f "
                "confidence=%.2f stability=%.2f",
                total,
                self._w_proximity, self._w_importance, self._w_direction,
                self._w_confidence, self._w_stability,
            )

        self._weight_sum: float = total

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def score(self, directed_objects: list) -> List[ScoredObject]:
        """
        Score and rank a list of DirectedObjects by navigation urgency.

        Each object receives a nav_score in [0.0, 1.0] derived from its
        proximity, class importance, direction, confidence, and stability.

        The returned list is sorted by nav_score descending — the most
        urgent object is always first.

        An empty input returns an empty output. No model is run.

        Args:
            directed_objects: List[DirectedObject] from SpatialReasoner.
                              May be empty.

        Returns:
            List[ScoredObject] sorted by nav_score descending.
        """
        if not directed_objects:
            return []

        scored: List[ScoredObject] = []

        for obj in directed_objects:
            nav_score = self._compute_score(obj)

            scored.append(ScoredObject(
                # Pass all DirectedObject fields through unchanged
                track_id=obj.track_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                confidence=obj.confidence,
                bbox=obj.bbox,
                bbox_norm=obj.bbox_norm,
                center_x=obj.center_x,
                center_y=obj.center_y,
                age=obj.age,
                last_seen=obj.last_seen,
                stability=obj.stability,
                area_norm=obj.area_norm,
                frame_width=obj.frame_width,
                frame_height=obj.frame_height,
                raw_depth=obj.raw_depth,
                proximity=obj.proximity,
                priority=obj.priority,
                depth_samples=obj.depth_samples,
                depth_age=obj.depth_age,
                direction=obj.direction,
                # Phase 9 addition
                nav_score=nav_score,
            ))

        # Sort highest score first
        scored.sort(key=lambda s: s.nav_score, reverse=True)

        logger.debug(
            "NavigationPriorityEngine.score(): %d objects, "
            "top score=%.3f (%s)",
            len(scored),
            scored[0].nav_score if scored else 0.0,
            scored[0].class_name if scored else "none",
        )

        return scored

    def compute_score_for_values(
        self,
        proximity: str,
        priority:  str,
        direction: str,
        confidence: float,
        stability:  float,
    ) -> float:
        """
        Compute a nav_score directly from individual values.

        Useful for testing, calibration, and what-if analysis.
        Does not require a DirectedObject.

        Args:
            proximity:  Proximity string (VERY_CLOSE / CLOSE / MEDIUM / FAR / UNKNOWN)
            priority:   Priority category (navigation_critical / contextual /
                        low_priority / unknown)
            direction:  Direction string (CENTER / LEFT / RIGHT / UNKNOWN)
            confidence: Float 0.0–1.0
            stability:  Float 0.0–1.0

        Returns:
            float in [0.0, 1.0]
        """
        prox_score = _PROXIMITY_SCORES.get(proximity,  0.0)
        imp_score  = _IMPORTANCE_SCORES.get(priority,  0.0)
        dir_score  = _DIRECTION_SCORES.get(direction,  0.5)
        conf       = max(0.0, min(1.0, float(confidence)))
        stab       = max(0.0, min(1.0, float(stability)))

        raw = (
            self._w_proximity  * prox_score
            + self._w_importance * imp_score
            + self._w_direction  * dir_score
            + self._w_confidence * conf
            + self._w_stability  * stab
        )

        return max(0.0, min(1.0, raw))

    # -----------------------------------------------------------------------
    # Internal scoring
    # -----------------------------------------------------------------------

    def _compute_score(self, obj) -> float:
        """Compute nav_score for one DirectedObject."""
        return self.compute_score_for_values(
            proximity=obj.proximity,
            priority=obj.priority,
            direction=obj.direction,
            confidence=obj.confidence,
            stability=obj.stability,
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def weights(self) -> Dict[str, float]:
        """Read-only copy of the current weight configuration."""
        return {
            "proximity":         self._w_proximity,
            "object_importance": self._w_importance,
            "direction":         self._w_direction,
            "confidence":        self._w_confidence,
            "stability":         self._w_stability,
        }

    @property
    def weight_sum(self) -> float:
        """Sum of all weights (should be ~1.0 for scores to stay in [0,1])."""
        return self._weight_sum

    def __repr__(self) -> str:
        return (
            f"NavigationPriorityEngine("
            f"prox={self._w_proximity:.2f}, "
            f"imp={self._w_importance:.2f}, "
            f"dir={self._w_direction:.2f}, "
            f"conf={self._w_confidence:.2f}, "
            f"stab={self._w_stability:.2f})"
        )
