"""
Phase 16 — Layer 2: Synthetic Integrated-Pipeline Evaluation Harness
=====================================================================
Phase: 16
Status: IMPLEMENTED

Deterministically evaluate pipeline STAGES AFTER detection using
CONSTRUCTED ground truth. Correct output is known exactly by construction.

HONESTY BOUNDARY: measures PIPELINE LOGIC CORRECTNESS on synthetic inputs.
NOT real-world accuracy/precision/recall/mAP/generalization.

Each row is tagged with correctness_type:
  "exact"     - input fully determines correct output (mismatch = real bug).
  "heuristic" - threshold-dependent behavior (configured estimate, reported
                for self-consistency, not physical ground truth).

Uses the REAL stage classes. No config/threshold/pipeline changes.
No camera / YOLO / depth model required.
Writes data/evaluation/phase16_pipeline_eval_<timestamp>.csv (NEW file).

Usage:
    .venv/Scripts/python.exe tests/evaluation/phase16_pipeline_eval.py
    .venv/Scripts/python.exe tests/evaluation/phase16_pipeline_eval.py --no-write
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from assistive_navigation.utils.config_loader import load_config
from assistive_navigation.detection.detector import Detection
from assistive_navigation.tracking.tracker import ObjectTracker
from assistive_navigation.depth.fusion import DepthFusion
from assistive_navigation.navigation.spatial import SpatialReasoner, Direction
from assistive_navigation.navigation.priority import NavigationPriorityEngine, ScoredObject
from assistive_navigation.navigation.temporal import TemporalConfirmationFilter
from assistive_navigation.alerts.alert_manager import AlertManager

FRAME_W = 640
FRAME_H = 480
_OUT_DIR = _PROJECT_ROOT / "data" / "evaluation"


# ---------------------------------------------------------------------------
# Synthetic object builders (mirror the real dataclass fields exactly)
# ---------------------------------------------------------------------------

def make_detection(
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.90,
    center_x: float = 0.50,
    center_y: float = 0.50,
    area_norm: float = 0.30,
    fw: int = FRAME_W,
    fh: int = FRAME_H,
) -> Detection:
    """Build a synthetic Detection with a bbox that yields the given
    center_x/center_y and area_norm (so the tracker + area gate see
    consistent geometry)."""
    # Derive a bbox with the requested area and center.
    # area_norm = (w*h)/(fw*fh); pick a square-ish box.
    box_area_px = area_norm * fw * fh
    side = max(2.0, box_area_px ** 0.5)
    cx_px = center_x * fw
    cy_px = center_y * fh
    x1 = cx_px - side / 2.0
    y1 = cy_px - side / 2.0
    x2 = cx_px + side / 2.0
    y2 = cy_px + side / 2.0
    return Detection(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        bbox_norm=(x1 / fw, y1 / fh, x2 / fw, y2 / fh),
        center_x=center_x,
        center_y=center_y,
        area_norm=area_norm,
        frame_width=fw,
        frame_height=fh,
    )


def make_scored(
    track_id: int = 1,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.90,
    center_x: float = 0.50,
    center_y: float = 0.50,
    area_norm: float = 0.30,
    stability: float = 0.9,
    nav_score: float = 0.75,
    proximity: str = "CLOSE",
    priority: str = "navigation_critical",
    direction: str = "CENTER",
    age: int = 10,
    last_seen: int = 0,
    raw_depth: float = 0.60,
    depth_age: int = 0,
    fw: int = FRAME_W,
    fh: int = FRAME_H,
) -> ScoredObject:
    """Build a synthetic ScoredObject (post-priority) for temporal/alert tests."""
    x1, y1 = (center_x - 0.1) * fw, (center_y - 0.1) * fh
    x2, y2 = (center_x + 0.1) * fw, (center_y + 0.1) * fh
    return ScoredObject(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=(x1, y1, x2, y2),
        bbox_norm=(x1 / fw, y1 / fh, x2 / fw, y2 / fh),
        center_x=center_x,
        center_y=center_y,
        age=age,
        last_seen=last_seen,
        stability=stability,
        area_norm=area_norm,
        frame_width=fw,
        frame_height=fh,
        raw_depth=raw_depth,
        proximity=proximity,
        priority=priority,
        depth_samples=50,
        depth_age=depth_age,
        direction=direction,
        nav_score=nav_score,
    )


def uniform_depth_map(value: float, fw: int = FRAME_W, fh: int = FRAME_H) -> np.ndarray:
    """A full-frame depth map filled with one relative value (0..1)."""
    return np.full((fh, fw), float(value), dtype=np.float32)


# ===========================================================================
# A. DIRECTION ACCURACY (exact) + boundaries
# ===========================================================================

def eval_direction(config: dict) -> List[dict]:
    """
    Evaluate SpatialReasoner.classify_direction over >=21 known positions.

    Ground truth is derived from the SAME half-open interval rule the code
    documents (LEFT [0,left_end), CENTER [left_end,right_start), RIGHT
    [right_start,1]). Positions strictly inside a zone are "exact". Positions
    exactly on a boundary are reported separately as "boundary" rows.
    """
    sr = SpatialReasoner(config)
    left_end = sr.left_zone_end
    right_start = sr.right_zone_start

    def expected(cx: float) -> str:
        c = max(0.0, min(1.0, cx))
        if c < left_end:
            return Direction.LEFT
        if c < right_start:
            return Direction.CENTER
        return Direction.RIGHT

    rows: List[dict] = []

    # 21-point grid across [0,1]
    grid = [i / 20.0 for i in range(21)]
    # Extra interior points to be safe away from boundaries
    grid += [0.02, 0.17, 0.33, 0.50, 0.66, 0.82, 0.98]

    for cx in grid:
        got = sr.classify_direction(cx)
        exp = expected(cx)
        is_boundary = abs(cx - left_end) < 1e-9 or abs(cx - right_start) < 1e-9
        rows.append({
            "group": "direction",
            "case": f"center_x={cx:.4f}",
            "correctness_type": "boundary" if is_boundary else "exact",
            "expected": exp,
            "got": got,
            "passed": (got == exp),
            "note": "half-open interval; boundary belongs to the right-hand zone",
        })

    # Explicit boundary cases (documented deterministic behavior)
    for cx, exp in [(left_end, Direction.CENTER), (right_start, Direction.RIGHT)]:
        got = sr.classify_direction(cx)
        rows.append({
            "group": "direction",
            "case": f"boundary center_x={cx:.4f}",
            "correctness_type": "boundary",
            "expected": exp,
            "got": got,
            "passed": (got == exp),
            "note": "boundary is deterministic by design",
        })

    # Out-of-range clamp + invalid
    rows.append({
        "group": "direction", "case": "center_x=-0.5 (clamp)",
        "correctness_type": "exact", "expected": Direction.LEFT,
        "got": sr.classify_direction(-0.5),
        "passed": sr.classify_direction(-0.5) == Direction.LEFT,
        "note": "clamped to 0.0 -> LEFT",
    })
    rows.append({
        "group": "direction", "case": "center_x=1.5 (clamp)",
        "correctness_type": "exact", "expected": Direction.RIGHT,
        "got": sr.classify_direction(1.5),
        "passed": sr.classify_direction(1.5) == Direction.RIGHT,
        "note": "clamped to 1.0 -> RIGHT",
    })
    rows.append({
        "group": "direction", "case": "center_x=NaN (invalid)",
        "correctness_type": "exact", "expected": Direction.UNKNOWN,
        "got": sr.classify_direction(float("nan")),
        "passed": sr.classify_direction(float("nan")) == Direction.UNKNOWN,
        "note": "non-finite -> UNKNOWN",
    })
    return rows


# ===========================================================================
# B. PROXIMITY BANDS (heuristic-threshold) + MONOTONICITY (exact ordering)
# ===========================================================================

_PROX_ORDER = {"UNKNOWN": -1, "FAR": 0, "MEDIUM": 1, "CLOSE": 2, "VERY_CLOSE": 3}


def eval_proximity(config: dict) -> List[dict]:
    """
    Evaluate proximity classification across multiple known depth values,
    plus the monotonicity property.

    Band membership is 'heuristic' (depends on configured thresholds
    very_close=0.75, close=0.55, medium=0.35 which are documented relative
    estimates). Monotonicity (closer depth never yields a farther label)
    is an 'exact' ordering property that must always hold.
    """
    fusion = DepthFusion(config)
    prox = config.get("proximity", {})
    t_vc = float(prox.get("very_close_threshold", 0.75))
    t_c = float(prox.get("close_threshold", 0.55))
    t_m = float(prox.get("medium_threshold", 0.35))

    def expected_band(v: float) -> str:
        if v > t_vc:
            return "VERY_CLOSE"
        if v > t_c:
            return "CLOSE"
        if v > t_m:
            return "MEDIUM"
        return "FAR"

    rows: List[dict] = []
    # Multiple values per band, safely away from thresholds
    test_values = [
        0.05, 0.15, 0.25, 0.34,        # FAR
        0.40, 0.45, 0.50, 0.54,        # MEDIUM
        0.58, 0.65, 0.70, 0.74,        # CLOSE
        0.80, 0.90, 0.99,              # VERY_CLOSE
    ]
    for v in test_values:
        got = fusion._classify(v)  # noqa: SLF001
        exp = expected_band(v)
        near_thresh = min(abs(v - t_vc), abs(v - t_c), abs(v - t_m)) < 0.02
        rows.append({
            "group": "proximity",
            "case": f"depth={v:.3f}",
            "correctness_type": "boundary" if near_thresh else "heuristic",
            "expected": exp,
            "got": got,
            "passed": (got == exp),
            "note": "band membership depends on configured relative thresholds",
        })

    # Monotonicity: as depth increases, band order must be non-decreasing.
    ladder = [0.05, 0.20, 0.34, 0.40, 0.54, 0.58, 0.74, 0.80, 0.99]
    prev_order = -99
    mono_ok = True
    for v in ladder:
        order = _PROX_ORDER[fusion._classify(v)]  # noqa: SLF001
        if order < prev_order:
            mono_ok = False
        prev_order = order
    rows.append({
        "group": "proximity",
        "case": "monotonicity (increasing depth -> non-decreasing band)",
        "correctness_type": "exact",
        "expected": "non-decreasing",
        "got": "non-decreasing" if mono_ok else "VIOLATED",
        "passed": mono_ok,
        "note": "ordering property must always hold",
    })
    return rows


def eval_proximity_end_to_end(config: dict) -> List[dict]:
    """
    Verify the FULL fusion path (fuse -> ROI sample -> smooth -> classify)
    produces the expected band for a uniform depth map. This exercises real
    ROI sampling and smoothing, not just the pure _classify threshold fn.
    Uses the real tracker to produce a TrackedObject first.
    """
    rows: List[dict] = []
    prox = config.get("proximity", {})
    t_vc = float(prox.get("very_close_threshold", 0.75))
    t_c = float(prox.get("close_threshold", 0.55))
    t_m = float(prox.get("medium_threshold", 0.35))

    def expected_band(v: float) -> str:
        if v > t_vc:
            return "VERY_CLOSE"
        if v > t_c:
            return "CLOSE"
        if v > t_m:
            return "MEDIUM"
        return "FAR"

    for v in [0.20, 0.45, 0.65, 0.90]:
        tracker = ObjectTracker(config)
        fusion = DepthFusion(config)
        det = make_detection(area_norm=0.30, center_x=0.5, center_y=0.5)
        dmap = uniform_depth_map(v)
        got = "UNKNOWN"
        # Run a few frames so the track establishes; depth constant each frame.
        for _ in range(6):
            tracked = tracker.update([det], FRAME_W, FRAME_H)
            fused = fusion.fuse(tracked, dmap, depth_age=0)
            if fused:
                got = fused[0].proximity
        exp = expected_band(v)
        rows.append({
            "group": "proximity_e2e",
            "case": f"uniform_depth={v:.2f}",
            "correctness_type": "heuristic",
            "expected": exp,
            "got": got,
            "passed": (got == exp),
            "note": "full fuse() path incl. ROI sampling + smoothing",
        })
    return rows


# ===========================================================================
# C + D. TEMPORAL CONFIRMATION (exact) + AREA GATE @ 0.15 (exact given config)
# ===========================================================================

def _fresh_temporal(config: dict) -> TemporalConfirmationFilter:
    return TemporalConfirmationFilter(config)


def eval_temporal_and_area(config: dict) -> List[dict]:
    rows: List[dict] = []
    cframes = int(config["temporal"]["confirmation_frames"])
    min_area = float(config["temporal"]["min_area_norm"])
    miss_tol = int(config["temporal"]["miss_tolerance"])

    # C1: single-frame phantom never confirms
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=0.30)
    c = tcf.update([obj])[0]
    rows.append({
        "group": "temporal", "case": "single-frame phantom",
        "correctness_type": "exact",
        "expected": "not confirmed", "got": ("confirmed" if c.is_confirmed else "not confirmed"),
        "passed": (c.is_confirmed is False),
        "note": "1 frame < confirmation_frames",
    })

    # C2: persistent object confirms at exactly confirmation_frames
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=0.30)
    confirmed_at = None
    for i in range(1, cframes + 3):
        c = tcf.update([obj])[0]
        if c.is_confirmed and confirmed_at is None:
            confirmed_at = i
    rows.append({
        "group": "temporal", "case": "persistent object confirmation frame",
        "correctness_type": "exact",
        "expected": f"confirms at frame {cframes}", "got": f"confirmed_at={confirmed_at}",
        "passed": (confirmed_at == cframes),
        "note": "state machine reaches CONFIRMED at confirmation_frames",
    })

    # C3: brief gap <= miss_tolerance resumes (does not reset to NEW)
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=0.30)
    for _ in range(4):
        tcf.update([obj])
    # gap of miss_tolerance frames (object absent)
    for _ in range(miss_tol):
        tcf.update([])
    c = tcf.update([obj])[0]
    rows.append({
        "group": "temporal", "case": f"gap<=miss_tolerance ({miss_tol}) resumes",
        "correctness_type": "exact",
        "expected": "count resumes (>1)", "got": f"count={c.consecutive_count}, state={c.confirmation_state}",
        "passed": (c.consecutive_count > 1),
        "note": "LOST state restores prior count on reappearance",
    })

    # C4: excessive gap > miss_tolerance expires (fresh start)
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=0.30)
    for _ in range(4):
        tcf.update([obj])
    for _ in range(miss_tol + 2):
        tcf.update([])
    c = tcf.update([obj])[0]
    rows.append({
        "group": "temporal", "case": f"gap>miss_tolerance expires",
        "correctness_type": "exact",
        "expected": "fresh start (count==1)", "got": f"count={c.consecutive_count}",
        "passed": (c.consecutive_count == 1),
        "note": "state deleted after miss_tolerance exceeded",
    })

    # D1: area exactly at genuine minimum (0.1612) still confirms at 0.15 gate
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=0.1612)
    last = None
    for _ in range(cframes):
        last = tcf.update([obj])[0]
    rows.append({
        "group": "area_gate", "case": f"area=0.1612 (genuine min) @ min_area_norm={min_area}",
        "correctness_type": "exact",
        "expected": "confirmed", "got": ("confirmed" if last.is_confirmed else "not confirmed"),
        "passed": (last.is_confirmed is True),
        "note": "genuine minimum area (Phase 15 evidence) passes the gate",
    })

    # D2: FP-profile area (0.0645 median) never confirms (hard fail)
    tcf = _fresh_temporal(config)
    fp = make_scored(track_id=1, area_norm=0.0645, class_name="person", center_x=0.79)
    last = None
    for _ in range(cframes + 4):
        last = tcf.update([fp])[0]
    rows.append({
        "group": "area_gate", "case": f"area=0.0645 (FP median) @ min_area_norm={min_area}",
        "correctness_type": "exact",
        "expected": "not confirmed + area_too_small", "got": f"confirmed={last.is_confirmed}, failures={last.gate_failures}",
        "passed": (last.is_confirmed is False and "area_too_small" in last.gate_failures),
        "note": "Phase 15 area gate hard-fails the dominant FP profile",
    })

    # D3: borderline just below gate suppressed
    tcf = _fresh_temporal(config)
    obj = make_scored(track_id=1, area_norm=min_area - 0.001)
    last = None
    for _ in range(cframes + 2):
        last = tcf.update([obj])[0]
    rows.append({
        "group": "area_gate", "case": f"area just below gate ({min_area-0.001:.3f})",
        "correctness_type": "exact",
        "expected": "not confirmed", "got": ("confirmed" if last.is_confirmed else "not confirmed"),
        "passed": (last.is_confirmed is False),
        "note": "hard area fail below threshold",
    })
    return rows


# ===========================================================================
# E. TRACKING PERSISTENCE / ID SWITCHES / SHORT-OCCLUSION COASTING
# ===========================================================================

def eval_tracking(config: dict) -> List[dict]:
    rows: List[dict] = []

    # E1: smoothly moving object keeps ONE track_id (no ID switch)
    tracker = ObjectTracker(config)
    seen_ids = set()
    id_per_frame = []
    for i in range(12):
        cx = 0.30 + i * 0.03  # smooth left->right drift
        det = make_detection(class_id=0, class_name="person",
                             center_x=cx, center_y=0.5, area_norm=0.30)
        tracked = tracker.update([det], FRAME_W, FRAME_H)
        if tracked:
            tid = tracked[0].track_id
            seen_ids.add(tid)
            id_per_frame.append(tid)
    # After warm-up (min_track_age), a single stable object should hold one id.
    stable_ids = set(id_per_frame[3:]) if len(id_per_frame) > 3 else set(id_per_frame)
    rows.append({
        "group": "tracking", "case": "smooth motion single-ID",
        "correctness_type": "heuristic",
        "expected": "1 stable id after warm-up", "got": f"stable_ids={sorted(stable_ids)}",
        "passed": (len(stable_ids) == 1),
        "note": "ByteTrack association under smooth motion (tracker heuristic)",
    })

    # E2: short occlusion (object absent a few frames within max_missed) -> id persists
    max_missed = int(config["tracking"]["max_missed_frames"])
    tracker = ObjectTracker(config)
    det = make_detection(center_x=0.5, center_y=0.5, area_norm=0.30)
    id_before = None
    for _ in range(6):
        tr = tracker.update([det], FRAME_W, FRAME_H)
        if tr:
            id_before = tr[0].track_id
    # brief occlusion: a couple of empty frames (< max_missed)
    gap = min(3, max_missed - 1) if max_missed > 1 else 1
    for _ in range(gap):
        tracker.update([], FRAME_W, FRAME_H)
    tr = tracker.update([det], FRAME_W, FRAME_H)
    id_after = tr[0].track_id if tr else None
    rows.append({
        "group": "tracking", "case": f"short occlusion ({gap} frames) coasting",
        "correctness_type": "heuristic",
        "expected": "same id persists", "got": f"before={id_before}, after={id_after}",
        "passed": (id_before is not None and id_after == id_before),
        "note": "track survives brief gap within max_missed_frames",
    })

    # E3: empty input -> zero tracks, no crash (exact)
    tracker = ObjectTracker(config)
    tr = tracker.update([], FRAME_W, FRAME_H)
    rows.append({
        "group": "tracking", "case": "empty input",
        "correctness_type": "exact",
        "expected": "0 tracks", "got": f"{len(tr)} tracks",
        "passed": (len(tr) == 0),
        "note": "no detections -> no tracks",
    })
    return rows


# ===========================================================================
# F. ALERT CORRECTNESS / TRIGGERS / TEXT / SELECTION / DEDUP / COUNT / LATENCY
# ===========================================================================

def _confirm(tcf, alert_mgr, objs, cycles):
    """Push objs through temporal+alert for N cycles; return list of AlertResult."""
    results = []
    for _ in range(cycles):
        confirmed = tcf.update(objs)
        results.append(alert_mgr.process(confirmed))
    return results


def eval_alerts(config: dict) -> List[dict]:
    rows: List[dict] = []
    cframes = int(config["temporal"]["confirmation_frames"])

    # F1: confirmed object -> exactly one alert with trigger new_object,
    # and pipeline-internal alert latency (frames from first eligible appearance
    # to alert_issued). Latency EXCLUDES TTS/audio playback time by construction
    # (AlertManager only enqueues text; we never call TTS here).
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    obj = make_scored(track_id=1, class_name="person", direction="CENTER",
                     proximity="CLOSE", nav_score=0.8, area_norm=0.30)
    first_alert_frame = None
    issued_reasons = []
    for i in range(1, cframes + 6):
        confirmed = tcf.update([obj])
        res = am.process(confirmed)
        if res.alert_issued:
            issued_reasons.append(res.trigger_reason)
            if first_alert_frame is None:
                first_alert_frame = i
    rows.append({
        "group": "alert", "case": "first alert trigger reason",
        "correctness_type": "exact",
        "expected": "new_object", "got": (issued_reasons[0] if issued_reasons else "none"),
        "passed": (bool(issued_reasons) and issued_reasons[0] == "new_object"),
        "note": "first alert for a newly confirmed track is new_object",
    })
    # latency: alert must occur exactly when confirmation is reached
    rows.append({
        "group": "alert", "case": "pipeline alert latency (frames, EXCLUDES TTS/audio)",
        "correctness_type": "exact",
        "expected": f"{cframes} frames", "got": f"{first_alert_frame} frames",
        "passed": (first_alert_frame == cframes),
        "note": "decision latency = confirmation_frames; audio playback NOT included",
    })

    # F2: alert text matches template for a standard alert
    #     standard: "{object} {direction}, {proximity}" with direction=ahead, proximity=close
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    obj = make_scored(track_id=5, class_name="person", direction="CENTER",
                     proximity="CLOSE", nav_score=0.8, area_norm=0.30)
    text = ""
    for _ in range(cframes):
        res = am.process(tcf.update([obj]))
        if res.alert_issued:
            text = res.alert_text
    rows.append({
        "group": "alert", "case": "alert text formatting",
        "correctness_type": "exact",
        "expected": "person ahead, close", "got": text,
        "passed": (text == "person ahead, close"),
        "note": "standard template with configured direction/proximity labels",
    })

    # F3: duplicate suppression - static unchanged object -> zero repeat alerts
    #     within cooldown. Run many cycles after first alert; count total alerts.
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    obj = make_scored(track_id=7, class_name="chair", direction="CENTER",
                     proximity="CLOSE", nav_score=0.7, area_norm=0.30)
    total_alerts = 0
    for _ in range(cframes + 40):  # well within 5s cooldown at test speed
        res = am.process(tcf.update([obj]))
        if res.alert_issued:
            total_alerts += 1
    rows.append({
        "group": "alert", "case": "duplicate suppression within cooldown",
        "correctness_type": "exact",
        "expected": "exactly 1 alert", "got": f"{total_alerts} alerts",
        "passed": (total_alerts == 1),
        "note": "unchanged static object: only the new_object alert, then suppressed",
    })

    # F4: highest nav_score selection when multiple confirmed objects
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    low = make_scored(track_id=10, class_name="bottle", direction="LEFT",
                     proximity="MEDIUM", nav_score=0.30, area_norm=0.30, center_x=0.2)
    high = make_scored(track_id=11, class_name="person", direction="CENTER",
                      proximity="VERY_CLOSE", nav_score=0.95, area_norm=0.30, center_x=0.5)
    selected = None
    for _ in range(cframes):
        res = am.process(tcf.update([low, high]))
        if res.alert_issued:
            selected = res.selected_track
    rows.append({
        "group": "alert", "case": "highest nav_score selection",
        "correctness_type": "exact",
        "expected": "track 11 (high score)", "got": f"track {selected}",
        "passed": (selected == 11),
        "note": "AlertManager selects max nav_score candidate",
    })

    # F5: proximity escalation triggers a fresh alert even within cooldown
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    near = make_scored(track_id=20, class_name="person", direction="CENTER",
                      proximity="MEDIUM", nav_score=0.6, area_norm=0.30)
    reasons = []
    for _ in range(cframes):
        res = am.process(tcf.update([near]))
        if res.alert_issued:
            reasons.append(res.trigger_reason)
    # now escalate proximity to VERY_CLOSE (same track)
    nearer = make_scored(track_id=20, class_name="person", direction="CENTER",
                        proximity="VERY_CLOSE", nav_score=0.9, area_norm=0.30)
    esc_reason = None
    for _ in range(3):
        res = am.process(tcf.update([nearer]))
        if res.alert_issued:
            esc_reason = res.trigger_reason
    rows.append({
        "group": "alert", "case": "proximity escalation trigger",
        "correctness_type": "exact",
        "expected": "proximity_escalation", "got": str(esc_reason),
        "passed": (esc_reason == "proximity_escalation"),
        "note": "moving strictly closer re-alerts despite cooldown",
    })
    return rows


# ===========================================================================
# G. MULTIPLE OBJECTS / PARTIAL SMALL-AREA / DIFFICULT BACKGROUND FP PROFILE
# ===========================================================================

def eval_scenarios(config: dict) -> List[dict]:
    rows: List[dict] = []
    cframes = int(config["temporal"]["confirmation_frames"])
    min_area = float(config["temporal"]["min_area_norm"])

    # G1: multiple objects all confirm independently and all appear
    tcf = TemporalConfirmationFilter(config)
    objs = [
        make_scored(track_id=1, class_name="person", center_x=0.2, area_norm=0.30),
        make_scored(track_id=2, class_name="chair", center_x=0.5, area_norm=0.30),
        make_scored(track_id=3, class_name="bottle", center_x=0.8, area_norm=0.30),
    ]
    last = []
    for _ in range(cframes):
        last = tcf.update(objs)
    confirmed_ids = sorted(o.track_id for o in last if o.is_confirmed)
    rows.append({
        "group": "multi_object", "case": "3 objects confirm independently",
        "correctness_type": "exact",
        "expected": "[1, 2, 3]", "got": str(confirmed_ids),
        "passed": (confirmed_ids == [1, 2, 3]),
        "note": "independent per-track temporal state",
    })

    # G2: partial / small-area object just ABOVE the gate confirms
    tcf = TemporalConfirmationFilter(config)
    small_ok = make_scored(track_id=1, area_norm=min_area + 0.01, class_name="bottle")
    last = None
    for _ in range(cframes):
        last = tcf.update([small_ok])[0]
    rows.append({
        "group": "partial", "case": f"small area just above gate ({min_area+0.01:.3f})",
        "correctness_type": "exact",
        "expected": "confirmed", "got": ("confirmed" if last.is_confirmed else "not confirmed"),
        "passed": (last.is_confirmed is True),
        "note": "small but valid object above 0.15 gate still confirms",
    })

    # G3: partial / small-area object just BELOW the gate is suppressed
    tcf = TemporalConfirmationFilter(config)
    small_no = make_scored(track_id=1, area_norm=min_area - 0.02, class_name="bottle")
    last = None
    for _ in range(cframes + 3):
        last = tcf.update([small_no])[0]
    rows.append({
        "group": "partial", "case": f"small area just below gate ({min_area-0.02:.3f})",
        "correctness_type": "exact",
        "expected": "not confirmed", "got": ("confirmed" if last.is_confirmed else "not confirmed"),
        "passed": (last.is_confirmed is False),
        "note": "small object below gate suppressed (Phase 15 behavior)",
    })

    # G4: difficult/background FP profile (area 0.0645, off-center 'person')
    #     must NEVER confirm and never produce an alert.
    tcf = TemporalConfirmationFilter(config)
    am = AlertManager(config, audio_queue=None)
    fp = make_scored(track_id=99, class_name="person", area_norm=0.0645,
                    center_x=0.79, center_y=0.5, nav_score=0.5, proximity="CLOSE")
    any_alert = False
    confirmed = False
    for _ in range(cframes + 20):
        conf = tcf.update([fp])
        if conf and conf[0].is_confirmed:
            confirmed = True
        if am.process(conf).alert_issued:
            any_alert = True
    rows.append({
        "group": "difficult_fp", "case": "background FP profile (area 0.0645)",
        "correctness_type": "exact",
        "expected": "never confirmed, no alert",
        "got": f"confirmed={confirmed}, any_alert={any_alert}",
        "passed": (confirmed is False and any_alert is False),
        "note": "dominant Phase 4 FP profile suppressed end-to-end by area gate",
    })
    return rows


# ===========================================================================
# Runner + CSV writer
# ===========================================================================

def run_all(config: Optional[dict] = None) -> List[dict]:
    """Run every Layer 2 evaluation and return the combined rows."""
    if config is None:
        config = load_config()
    rows: List[dict] = []
    rows += eval_direction(config)
    rows += eval_proximity(config)
    rows += eval_proximity_end_to_end(config)
    rows += eval_temporal_and_area(config)
    rows += eval_tracking(config)
    rows += eval_alerts(config)
    rows += eval_scenarios(config)
    return rows


def summarize(rows: List[dict]) -> dict:
    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    exact = [r for r in rows if r["correctness_type"] == "exact"]
    exact_passed = sum(1 for r in exact if r["passed"])
    heuristic = [r for r in rows if r["correctness_type"] == "heuristic"]
    heuristic_passed = sum(1 for r in heuristic if r["passed"])
    boundary = [r for r in rows if r["correctness_type"] == "boundary"]
    boundary_passed = sum(1 for r in boundary if r["passed"])
    by_group: Dict[str, dict] = {}
    for r in rows:
        g = by_group.setdefault(r["group"], {"total": 0, "passed": 0})
        g["total"] += 1
        g["passed"] += 1 if r["passed"] else 0
    return {
        "total": total, "passed": passed,
        "exact_total": len(exact), "exact_passed": exact_passed,
        "heuristic_total": len(heuristic), "heuristic_passed": heuristic_passed,
        "boundary_total": len(boundary), "boundary_passed": boundary_passed,
        "by_group": by_group,
    }


_CSV_COLUMNS = ["group", "case", "correctness_type", "expected", "got", "passed", "note"]


def write_csv(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _CSV_COLUMNS})


def print_summary(rows: List[dict], summary: dict) -> None:
    print("=" * 72)
    print("  PHASE 16 — LAYER 2: SYNTHETIC PIPELINE EVALUATION")
    print("=" * 72)
    print("  Synthetic ground truth (by construction). Measures pipeline LOGIC")
    print("  correctness, NOT real-world accuracy/precision/recall/mAP.")
    print("-" * 72)
    for g, s in sorted(summary["by_group"].items()):
        print(f"    {g:<16} {s['passed']:>3}/{s['total']:<3} passed")
    print("-" * 72)
    print(f"    EXACT (deterministic)  : {summary['exact_passed']}/{summary['exact_total']}")
    print(f"    BOUNDARY (deterministic): {summary['boundary_passed']}/{summary['boundary_total']}")
    print(f"    HEURISTIC (threshold)  : {summary['heuristic_passed']}/{summary['heuristic_total']}")
    print(f"    TOTAL                  : {summary['passed']}/{summary['total']}")
    print("-" * 72)
    fails = [r for r in rows if not r["passed"]]
    if fails:
        print("    FAILURES:")
        for r in fails:
            print(f"      [{r['group']}] {r['case']}: expected={r['expected']} got={r['got']}")
    else:
        print("    All Layer 2 checks passed.")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 16 Layer 2 synthetic pipeline evaluation (deterministic)."
    )
    parser.add_argument("--no-write", action="store_true",
                        help="Print results but do not write the CSV.")
    args = parser.parse_args()

    rows = run_all()
    summary = summarize(rows)
    print_summary(rows, summary)

    if not args.no_write:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _OUT_DIR / f"phase16_pipeline_eval_{ts}.csv"
        write_csv(rows, out)
        print(f"\n  Results written to: {out}")
        print("  (No baseline artifacts modified.)")


if __name__ == "__main__":
    main()
