"""
Unit Tests — Phase 16 Final Evaluation
=======================================
Phase: 16
Status: IMPLEMENTED

Deterministic assertions for the Phase 16 three-layer evaluation:

  LAYER 1 (detection-level, offline): the Phase 15 area-gate replay numbers
     reproduce EXACTLY from the existing Phase 4 CSV (read-only).
  LAYER 2 (synthetic pipeline): every by-construction correctness check in
     phase16_pipeline_eval passes; exact vs heuristic vs boundary are
     distinguished.
  LAYER 3 (hardware): perf-CSV analysis math is correct on a synthetic CSV
     (no real hardware needed); NOT-RUN handling is honest.
  GUARDS: Phase 15 config value (min_area_norm=0.15) and confidence (0.35)
     are unchanged; Phase 16 created no threshold drift.

HONESTY: Layer 2 correctness is BY CONSTRUCTION (synthetic ground truth).
These tests validate pipeline LOGIC, not real-world detection accuracy.
No precision/recall/mAP/FN-rate/generalization is asserted anywhere.

Run:
    .venv/Scripts/python.exe -m pytest tests/unit/test_phase16.py -v
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
sys.path.insert(0, str(_PROJECT_ROOT / "tests" / "evaluation"))

import phase16_pipeline_eval as p16     # noqa: E402
import phase16_final_eval as p16final   # noqa: E402


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    from assistive_navigation.utils.config_loader import load_config
    return load_config()


@pytest.fixture(scope="module")
def layer2_rows(config):
    return p16.run_all(config)


@pytest.fixture(scope="module")
def layer2_summary(layer2_rows):
    return p16.summarize(layer2_rows)


@pytest.fixture(scope="module")
def l1_detection():
    csv_path = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
    if not csv_path.exists():
        pytest.skip("phase4_results.csv not present")
    return p16final.layer1_detection(csv_path)


# ===========================================================================
# GROUP 1 — Layer 2 overall: every synthetic check passes
# ===========================================================================

class TestLayer2Overall:

    def test_all_layer2_checks_pass(self, layer2_rows):
        fails = [r for r in layer2_rows if not r["passed"]]
        assert not fails, "Layer 2 failures: " + "; ".join(
            f"[{r['group']}] {r['case']} exp={r['expected']} got={r['got']}"
            for r in fails
        )

    def test_has_minimum_check_count(self, layer2_rows):
        # Should be a substantial number of deterministic checks.
        assert len(layer2_rows) >= 60

    def test_correctness_types_present(self, layer2_rows):
        types = {r["correctness_type"] for r in layer2_rows}
        # We must clearly distinguish exact vs heuristic (boundary optional).
        assert "exact" in types
        assert "heuristic" in types

    def test_every_row_has_schema(self, layer2_rows):
        for r in layer2_rows:
            for key in ("group", "case", "correctness_type",
                        "expected", "got", "passed", "note"):
                assert key in r


# ===========================================================================
# GROUP 2 — Direction accuracy (exact) + boundaries
# ===========================================================================

class TestDirection:

    def test_direction_grid_all_pass(self, config):
        rows = p16.eval_direction(config)
        assert all(r["passed"] for r in rows)

    def test_direction_grid_has_21_plus_positions(self, config):
        rows = p16.eval_direction(config)
        # At least 21 non-boundary grid rows required.
        exact_rows = [r for r in rows if r["correctness_type"] == "exact"]
        assert len(exact_rows) >= 21

    def test_boundaries_reported_separately(self, config):
        rows = p16.eval_direction(config)
        boundary = [r for r in rows if r["correctness_type"] == "boundary"]
        assert len(boundary) >= 2
        assert all(r["passed"] for r in boundary)


# ===========================================================================
# GROUP 3 — Proximity bands + monotonicity
# ===========================================================================

class TestProximity:

    def test_proximity_bands_all_pass(self, config):
        rows = p16.eval_proximity(config)
        assert all(r["passed"] for r in rows)

    def test_monotonicity_holds(self, config):
        rows = p16.eval_proximity(config)
        mono = [r for r in rows if "monotonicity" in r["case"]]
        assert len(mono) == 1
        assert mono[0]["passed"] is True
        assert mono[0]["correctness_type"] == "exact"

    def test_proximity_end_to_end_all_pass(self, config):
        rows = p16.eval_proximity_end_to_end(config)
        assert all(r["passed"] for r in rows)
        assert len(rows) >= 4


# ===========================================================================
# GROUP 4 — Temporal + area gate @ 0.15
# ===========================================================================

class TestTemporalArea:

    def test_temporal_and_area_all_pass(self, config):
        rows = p16.eval_temporal_and_area(config)
        assert all(r["passed"] for r in rows), [
            (r["case"], r["got"]) for r in rows if not r["passed"]
        ]

    def test_fp_profile_hard_fails_area_gate(self, config):
        rows = p16.eval_temporal_and_area(config)
        fp = [r for r in rows if "FP median" in r["case"]]
        assert len(fp) == 1 and fp[0]["passed"]

    def test_genuine_min_area_confirms(self, config):
        rows = p16.eval_temporal_and_area(config)
        gm = [r for r in rows if "genuine min" in r["case"]]
        assert len(gm) == 1 and gm[0]["passed"]


# ===========================================================================
# GROUP 5 — Tracking
# ===========================================================================

class TestTracking:

    def test_tracking_all_pass(self, config):
        rows = p16.eval_tracking(config)
        assert all(r["passed"] for r in rows), [
            (r["case"], r["got"]) for r in rows if not r["passed"]
        ]

    def test_empty_input_zero_tracks_is_exact(self, config):
        rows = p16.eval_tracking(config)
        e = [r for r in rows if r["case"] == "empty input"]
        assert len(e) == 1
        assert e[0]["correctness_type"] == "exact"
        assert e[0]["passed"]


# ===========================================================================
# GROUP 6 — Alerts
# ===========================================================================

class TestAlerts:

    def test_alerts_all_pass(self, config):
        rows = p16.eval_alerts(config)
        assert all(r["passed"] for r in rows), [
            (r["case"], r["got"]) for r in rows if not r["passed"]
        ]

    def test_duplicate_suppression_single_alert(self, config):
        rows = p16.eval_alerts(config)
        dup = [r for r in rows if "duplicate suppression" in r["case"]]
        assert len(dup) == 1 and dup[0]["passed"]

    def test_latency_excludes_tts_labeled(self, config):
        rows = p16.eval_alerts(config)
        lat = [r for r in rows if "alert latency" in r["case"]]
        assert len(lat) == 1
        # The label must make the TTS exclusion explicit.
        assert "EXCLUDES TTS" in lat[0]["case"] or "audio" in lat[0]["note"]
        assert lat[0]["passed"]

    def test_highest_nav_score_selection(self, config):
        rows = p16.eval_alerts(config)
        sel = [r for r in rows if "highest nav_score" in r["case"]]
        assert len(sel) == 1 and sel[0]["passed"]


# ===========================================================================
# GROUP 7 — Multi / partial / difficult FP profile
# ===========================================================================

class TestScenarios:

    def test_scenarios_all_pass(self, config):
        rows = p16.eval_scenarios(config)
        assert all(r["passed"] for r in rows), [
            (r["case"], r["got"]) for r in rows if not r["passed"]
        ]

    def test_difficult_fp_never_confirms_no_alert(self, config):
        rows = p16.eval_scenarios(config)
        d = [r for r in rows if r["group"] == "difficult_fp"]
        assert len(d) == 1
        assert d[0]["correctness_type"] == "exact"
        assert d[0]["passed"]


# ===========================================================================
# GROUP 8 — Layer 1 replay reproduction (exact numbers)
# ===========================================================================

class TestLayer1Replay:

    def test_before_after_reproduce_phase15(self, l1_detection):
        l1 = l1_detection
        b = l1["replay_before_0_02"]
        a = l1["replay_after_0_15"]
        assert b["fp_suppressed"] == 72
        assert b["fp_total"] == 303
        assert b["gen_retained"] == 114
        assert a["fp_suppressed"] == 215
        assert a["gen_retained"] == 114
        assert a["gen_lost"] == 0

    def test_per_class_retention_full(self, l1_detection):
        pc = l1_detection["per_class_retention_0_15"]
        for cls in ("person", "chair", "laptop", "bottle"):
            assert pc[cls]["lost"] == 0
            assert pc[cls]["retained"] == pc[cls]["total"]

    def test_fp_accepted_total_303(self, l1_detection):
        assert l1_detection["fp_accepted_total"] == 303
        assert l1_detection["fp_accepted_by_scenario"]["empty_scene_fp_rate"] == 203

    def test_wrong_class_observations_reported(self, l1_detection):
        # FP-3 mislabels must be reported (not zero), documented as unsolved.
        assert l1_detection["wrong_class_observations"] > 0

    def test_no_detection_known_objects_reported(self, l1_detection):
        nd = l1_detection["no_detection_observations_known_objects"]
        # Scene-specific observed misses exist for known-object scenarios.
        assert isinstance(nd, dict)
        assert len(nd) > 0


# ===========================================================================
# GROUP 9 — Layer 3 perf analysis math (synthetic CSV, no hardware)
# ===========================================================================

class TestLayer3PerfMath:

    def _write_perf(self, tmp_path, rows):
        import csv as _csv
        p = tmp_path / "perf.csv"
        cols = ["frame_num", "total_ms", "detect_ms", "depth_ms", "depth_ran",
                "detections_raw", "detections_acc", "tracks_active",
                "confirmed_count", "alert_issued",
                "cpu_percent_system", "rss_mb_process"]
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return p

    def test_fps_stats_correct(self, tmp_path):
        # total_ms of 100 -> 10 FPS; 50 -> 20 FPS; 200 -> 5 FPS
        rows = []
        for i, (tm, dr) in enumerate([(100, "False"), (50, "True"), (200, "False")], 1):
            rows.append({
                "frame_num": i, "total_ms": tm, "detect_ms": 40, "depth_ms": (120 if dr == "True" else 0),
                "depth_ran": dr, "detections_raw": 1, "detections_acc": 1,
                "tracks_active": 1, "confirmed_count": 0, "alert_issued": "False",
                "cpu_percent_system": 30.0, "rss_mb_process": 420.0,
            })
        p = self._write_perf(tmp_path, rows)
        a = p16final.analyze_perf_csv(p)
        assert a["available"] is True
        assert a["fps"]["min"] == 5.0
        assert a["fps"]["max"] == 20.0
        # depth avg only over frames where depth ran (one frame, 120ms)
        assert a["depth_ms_when_ran"]["mean"] == 120.0
        assert a["depth_inference_count"] == 1

    def test_missing_csv_marked_not_run(self, tmp_path):
        a = p16final.analyze_perf_csv(tmp_path / "nope.csv")
        assert a["available"] is False
        assert "NOT RUN" in a["note"]

    def test_capture_copies_without_modifying_source(self, tmp_path):
        src = self._write_perf(tmp_path, [{
            "frame_num": 1, "total_ms": 80, "detect_ms": 40, "depth_ms": 0,
            "depth_ran": "False", "detections_raw": 0, "detections_acc": 0,
            "tracks_active": 0, "confirmed_count": 0, "alert_issued": "False",
            "cpu_percent_system": 0.0, "rss_mb_process": 400.0,
        }])
        src_before = src.read_text(encoding="utf-8")
        dest = p16final.capture_perf_csv(src, timestamp="unittest_ts")
        try:
            assert dest.exists()
            # source unchanged
            assert src.read_text(encoding="utf-8") == src_before
            assert "phase16_perf_unittest_ts" in dest.name
        finally:
            if dest.exists():
                dest.unlink()

    def test_fps_baseline_comparison_never_fails(self):
        # Low FPS must NOT raise / must return an informative string.
        low = p16final.compare_fps_to_baseline({"mean": 5.0, "median": 5.0, "min": 4.0, "max": 6.0})
        assert "not a phase failure" in low.lower()
        ok = p16final.compare_fps_to_baseline({"mean": 12.0, "median": 12.0, "min": 11.0, "max": 13.0})
        assert "within/above baseline" in ok


# ===========================================================================
# GROUP 10 — Baseline / config guards (Phase 16 changes nothing it must not)
# ===========================================================================

class TestBaselineGuards:

    def test_min_area_norm_still_0_15(self, config):
        assert config["temporal"]["min_area_norm"] == pytest.approx(0.15)

    def test_confidence_threshold_still_0_35(self, config):
        assert config["detection"]["confidence_threshold"] == pytest.approx(0.35)

    def test_confirmation_frames_still_8(self, config):
        assert config["temporal"]["confirmation_frames"] == 8

    def test_depth_model_unchanged(self, config):
        assert config["depth"]["model_name"] == "depth_anything_v2_small"

    def test_detector_model_unchanged(self, config):
        assert config["detection"]["model_name"] == "yolo11n"
