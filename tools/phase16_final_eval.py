"""
Phase 16 — Final Evaluation Orchestrator (Layer 1 + Layer 3 + Report)
======================================================================
Phase: 16
Status: IMPLEMENTED

Combines the three-layer final evaluation into one auditable report:

  LAYER 1 — Detection-level (OFFLINE, reuses existing Phase 4 CSV read-only):
    * Reproduces the Phase 15 area-gate replay numbers (0.02 vs 0.15).
    * Reports genuine-object retention by class.
    * Reports accepted false positives (empty-scene family).
    * Reports wrong-class / class-confusion observation counts.
    * Reports observed no-detection counts for known-object scenarios.
    * Does NOT claim precision/recall/FN-rate/mAP/accuracy/generalization.

  LAYER 2 — Synthetic pipeline (delegated to phase16_pipeline_eval.run_all):
    * Deterministic, by-construction correctness of pipeline stages.

  LAYER 3 — Hardware / live camera (OPTIONAL, operator-run):
    * Captures >=3 separate 60s performance runs.
    * FPS mean/median/min/max, detector/depth/total latency, depth
      inference count + interval adherence, system CPU, process RSS.
    * Live empty-scene FP run (60s).
    * Copies each completed performance.csv IMMEDIATELY to a timestamped
      data/evaluation/phase16_perf_<ts>.csv (because the live log is
      overwritten by every run).
    * If a hardware run is not executed, it is explicitly marked NOT RUN.
      This tool NEVER fabricates hardware numbers.

OUTPUT (all NEW, timestamped; nothing overwritten):
    data/evaluation/phase16_report_<ts>.txt

BASELINES PRESERVED: phase4_results.csv, phase4_report.txt, Phase 15
analysis outputs and config are never modified by this tool.

Usage:
    # Offline layers 1+2 and write the report (no hardware):
    .venv/Scripts/python.exe tools/phase16_final_eval.py

    # Analyse an already-captured performance CSV (e.g. after a live run):
    .venv/Scripts/python.exe tools/phase16_final_eval.py --perf-csv logs/performance.csv

    # Capture a live perf CSV to a timestamped Phase 16 file (after you have
    # run `python -m assistive_navigation --headless` for ~60s):
    .venv/Scripts/python.exe tools/phase16_final_eval.py --capture-perf logs/performance.csv
"""

import argparse
import csv
import shutil
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "tools"))
sys.path.insert(0, str(_PROJECT_ROOT / "tests" / "evaluation"))

import phase15_fp_analysis as p15          # Layer 1 evidence base (read-only)
import phase16_pipeline_eval as p16layer2  # Layer 2 synthetic harness

_EVAL_DIR = _PROJECT_ROOT / "data" / "evaluation"
_PHASE4_CSV = _EVAL_DIR / "phase4_results.csv"

# Phase 13-14 baseline FPS range (measured), for comparison only.
BASELINE_FPS_LOW = 11.6
BASELINE_FPS_HIGH = 12.5


# ===========================================================================
# LAYER 1 — Detection-level (offline, read-only reuse of Phase 4 CSV)
# ===========================================================================

def layer1_detection(csv_path: Path = _PHASE4_CSV) -> dict:
    """
    Compute Layer 1 detection-level metrics from the existing Phase 4 CSV.
    Reuses phase15_fp_analysis for the audited area-gate replay.
    READ-ONLY. Returns a structured dict (no claims of precision/recall/mAP).
    """
    result: dict = {"available": csv_path.exists()}
    if not csv_path.exists():
        result["note"] = "phase4_results.csv not present — Layer 1 NOT RUN."
        return result

    rows = p15.load_accepted_rows(csv_path)
    sweep = p15.sweep_thresholds(rows, [0.02, 0.15])

    def at(thr):
        return next(s for s in sweep if abs(s["threshold"] - thr) < 1e-9)

    before = at(0.02)
    after = at(0.15)

    # Genuine-object retention by class at the calibrated gate (0.15)
    per_class = p15.per_class_genuine_retention(rows, 0.15)

    # Accepted false positives (empty-scene family) — count by scenario
    fp_rows = [r for r in rows if r.is_false_positive]
    fp_by_scenario: Dict[str, int] = {}
    for r in fp_rows:
        fp_by_scenario[r.scenario] = fp_by_scenario.get(r.scenario, 0) + 1

    # Wrong-class / class-confusion observations: count directly from the CSV
    wrong_class = _count_eval_result(csv_path, "wrong_class_observation")
    no_detection = _count_no_detection_for_known_objects(csv_path)

    result.update({
        "replay_before_0_02": before,
        "replay_after_0_15": after,
        "per_class_retention_0_15": per_class,
        "fp_accepted_total": len(fp_rows),
        "fp_accepted_by_scenario": fp_by_scenario,
        "wrong_class_observations": wrong_class,
        "no_detection_observations_known_objects": no_detection,
    })
    return result


def _count_eval_result(csv_path: Path, eval_result: str) -> int:
    """Count CSV rows with a given eval_result code (read-only)."""
    n = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("eval_result") or "").strip() == eval_result:
                n += 1
    return n


# Known-object (true-positive / occlusion / multi) scenarios where a
# no-detection observation is a genuine observed miss for that scene.
_KNOWN_OBJECT_SCENARIOS = {
    "person_standing", "chair_visible", "dining_table_visible",
    "laptop_visible", "bottle_visible", "backpack_visible",
    "partial_person", "partial_chair", "edge_of_frame_bottle",
    "person_and_chair", "bottle_laptop_backpack",
}


def _count_no_detection_for_known_objects(csv_path: Path) -> Dict[str, int]:
    """
    Per-scenario 'no_detection_observation' counts for scenarios where a
    known object was declared present. Reported as SCENE-SPECIFIC observed
    misses — explicitly NOT a generalized false-negative rate.
    """
    counts: Dict[str, int] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scn = (row.get("scenario") or "").strip()
            if scn in _KNOWN_OBJECT_SCENARIOS and \
               (row.get("eval_result") or "").strip() == "no_detection_observation":
                counts[scn] = counts.get(scn, 0) + 1
    return counts


# ===========================================================================
# LAYER 3 — Hardware / live-camera (optional; never fabricated)
# ===========================================================================

_PERF_NUMERIC = ["total_ms", "detect_ms", "depth_ms"]


def capture_perf_csv(source: Path, timestamp: Optional[str] = None) -> Path:
    """
    Copy a completed performance.csv to a NEW timestamped Phase 16 file so it
    is not lost when the live log is overwritten by the next run.
    Returns the destination path. Does NOT modify the source.
    """
    if not source.exists():
        raise FileNotFoundError(f"performance CSV not found: {source}")
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _EVAL_DIR / f"phase16_perf_{ts}.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def analyze_perf_csv(perf_csv: Path) -> dict:
    """
    Compute FPS + latency + resource statistics from a performance CSV
    (PerfMonitor format). FPS is derived per-frame as 1000/total_ms and
    aggregated (mean/median/min/max). depth_ms is averaged only over frames
    where depth actually ran (depth_ran==True).

    CPU is SYSTEM-WIDE (cpu_percent_system column); RSS is process memory.
    Returns a dict; marks NOT RUN if the file is missing/empty.
    """
    if not perf_csv.exists():
        return {"available": False, "note": f"{perf_csv} not present — NOT RUN"}

    total_ms: List[float] = []
    detect_ms: List[float] = []
    depth_ms_ran: List[float] = []
    cpu_sys: List[float] = []
    rss: List[float] = []
    depth_ran_count = 0
    n = 0

    with open(perf_csv, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            try:
                t = float(row.get("total_ms") or 0)
                if t > 0:
                    total_ms.append(t)
                d = float(row.get("detect_ms") or 0)
                if d > 0:
                    detect_ms.append(d)
                ran = str(row.get("depth_ran", "")).strip().lower() == "true"
                if ran:
                    depth_ran_count += 1
                    dm = float(row.get("depth_ms") or 0)
                    if dm > 0:
                        depth_ms_ran.append(dm)
                c = float(row.get("cpu_percent_system") or 0)
                cpu_sys.append(c)
                r = float(row.get("rss_mb_process") or 0)
                if r > 0:
                    rss.append(r)
            except (ValueError, TypeError):
                continue

    if not total_ms:
        return {"available": False, "note": f"{perf_csv} has no usable rows — NOT RUN"}

    fps = [1000.0 / t for t in total_ms]

    def stats(vals):
        if not vals:
            return None
        return {
            "mean": round(statistics.mean(vals), 2),
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
        }

    return {
        "available": True,
        "source": str(perf_csv),
        "frames": n,
        "fps": stats(fps),
        "detect_ms": stats(detect_ms),
        "depth_ms_when_ran": stats(depth_ms_ran),
        "total_ms": stats(total_ms),
        "cpu_percent_system": stats(cpu_sys),
        "rss_mb_process": stats(rss),
        "depth_inference_count": depth_ran_count,
        "depth_interval_observed": round(n / depth_ran_count, 2) if depth_ran_count else None,
    }


def compare_fps_to_baseline(fps_stats: Optional[dict]) -> str:
    """Compare measured FPS mean to the Phase 13-14 baseline range. Never fails."""
    if not fps_stats:
        return "NOT RUN"
    mean = fps_stats["mean"]
    if mean >= BASELINE_FPS_LOW:
        return (f"mean {mean} FPS >= baseline low {BASELINE_FPS_LOW} "
                f"(baseline range {BASELINE_FPS_LOW}-{BASELINE_FPS_HIGH}) — within/above baseline")
    return (f"mean {mean} FPS below baseline low {BASELINE_FPS_LOW}; "
            f"a single run is NOT a phase failure — repeat before concluding")


# ===========================================================================
# REPORT
# ===========================================================================

def build_report(
    layer1: dict,
    layer2_rows: List[dict],
    layer2_summary: dict,
    perf_analyses: List[dict],
    live_fp: Optional[dict],
) -> str:
    L: List[str] = []
    add = L.append
    add("=" * 74)
    add("  PHASE 16 — FINAL SYSTEM EVALUATION REPORT")
    add("=" * 74)
    add(f"    Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add("    Scope     : complete assistive navigation system (research prototype)")
    add("")
    add("    HONESTY NOTICE:")
    add("      - No labelled dataset -> NO precision/recall/FN-rate/mAP/accuracy")
    add("        and NO generalization claims.")
    add("      - Layer 2 uses SYNTHETIC ground truth (by construction): it")
    add("        validates pipeline LOGIC, not real-world detection quality.")
    add("      - Depth is RELATIVE, not metric -> proximity is ordinal.")
    add("      - Single environment / camera / lighting.")
    add("      - Alert latency is PIPELINE-INTERNAL and EXCLUDES TTS/audio time.")
    add("=" * 74)

    # ---- Layer 1 ----
    add("  LAYER 1 — DETECTION-LEVEL (offline, existing Phase 4 CSV, read-only)")
    add("-" * 74)
    if not layer1.get("available"):
        add(f"    NOT RUN — {layer1.get('note', 'phase4_results.csv missing')}")
    else:
        b = layer1["replay_before_0_02"]
        a = layer1["replay_after_0_15"]
        add("    Phase 15 area-gate replay reproduction (audited from raw CSV):")
        add(f"      before min_area_norm=0.02: FP suppressed {b['fp_suppressed']}/{b['fp_total']} "
            f"({b['fp_suppression_pct']:.1f}%), genuine retained {b['gen_retained']}/{b['gen_total']}")
        add(f"      after  min_area_norm=0.15: FP suppressed {a['fp_suppressed']}/{a['fp_total']} "
            f"({a['fp_suppression_pct']:.1f}%), genuine retained {a['gen_retained']}/{a['gen_total']} "
            f"(lost {a['gen_lost']})")
        add("    Genuine-object retention by class @ 0.15:")
        for cls in sorted(layer1["per_class_retention_0_15"].keys()):
            c = layer1["per_class_retention_0_15"][cls]
            add(f"      {cls:<10}: retained {c['retained']}/{c['total']} (lost {c['lost']})")
        add(f"    Accepted false positives (empty-scene family): {layer1['fp_accepted_total']}")
        for scn, cnt in sorted(layer1["fp_accepted_by_scenario"].items()):
            add(f"      {scn:<22}: {cnt}")
        add(f"    Wrong-class / class-confusion observations: {layer1['wrong_class_observations']}")
        add("      (FP-3 mislabels — NOT addressed by the area gate; documented limitation)")
        add("    Observed no-detection counts for KNOWN-object scenarios")
        add("      (scene-specific observed misses; NOT a generalized FN rate):")
        nd = layer1["no_detection_observations_known_objects"]
        if nd:
            for scn in sorted(nd.keys()):
                add(f"      {scn:<22}: {nd[scn]}")
        else:
            add("      (none)")
    add("=" * 74)

    # ---- Layer 2 ----
    add("  LAYER 2 — SYNTHETIC INTEGRATED PIPELINE (deterministic, by construction)")
    add("-" * 74)
    for g, s in sorted(layer2_summary["by_group"].items()):
        add(f"    {g:<16} {s['passed']:>3}/{s['total']:<3} passed")
    add("-" * 74)
    add(f"    EXACT (deterministic)   : {layer2_summary['exact_passed']}/{layer2_summary['exact_total']}")
    add(f"    BOUNDARY (deterministic): {layer2_summary['boundary_passed']}/{layer2_summary['boundary_total']}")
    add(f"    HEURISTIC (threshold)   : {layer2_summary['heuristic_passed']}/{layer2_summary['heuristic_total']}")
    add(f"    TOTAL                   : {layer2_summary['passed']}/{layer2_summary['total']}")
    fails = [r for r in layer2_rows if not r["passed"]]
    if fails:
        add("    FAILURES:")
        for r in fails:
            add(f"      [{r['group']}] {r['case']}: expected={r['expected']} got={r['got']}")
    else:
        add("    All Layer 2 checks passed.")
    add("=" * 74)

    # ---- Layer 3 ----
    add("  LAYER 3 — HARDWARE / LIVE CAMERA")
    add("-" * 74)
    ran = [p for p in perf_analyses if p.get("available")]
    if not ran:
        add("    PERFORMANCE RUNS: NOT RUN")
        add("      No timestamped Phase 16 performance CSV was provided/captured.")
        add("      To run: `python -m assistive_navigation --headless` (~60s), then")
        add("      `python tools/phase16_final_eval.py --capture-perf logs/performance.csv`")
        add("      Repeat >=3 times. This tool does NOT fabricate hardware numbers.")
    else:
        add(f"    PERFORMANCE RUNS: {len(ran)} run(s) analysed")
        for i, p in enumerate(ran, 1):
            add(f"    Run {i} ({p['source']}, {p['frames']} frames):")
            add(f"      FPS        : {p['fps']}")
            add(f"      detect_ms  : {p['detect_ms']}")
            add(f"      depth_ms*  : {p['depth_ms_when_ran']}  (*only frames where depth ran)")
            add(f"      total_ms   : {p['total_ms']}")
            add(f"      CPU(sys)%  : {p['cpu_percent_system']}  (SYSTEM-WIDE, not per-process)")
            add(f"      RSS(MB)    : {p['rss_mb_process']}  (process memory)")
            add(f"      depth inferences: {p['depth_inference_count']}  "
                f"observed interval ~{p['depth_interval_observed']} frames")
            add(f"      vs baseline: {compare_fps_to_baseline(p['fps'])}")
    add("")
    if live_fp and live_fp.get("available"):
        add(f"    LIVE EMPTY-SCENE FP RUN: {live_fp['source']}")
        add(f"      accepted/min = {live_fp.get('accepted_per_min', 'n/a')} "
            f"(Phase 4 pre-temporal baseline was 202.9/min; full-pipeline value differs)")
        add("      NOTE: live FP is a MEASUREMENT, not a pass/fail criterion.")
    else:
        add("    LIVE EMPTY-SCENE FP RUN: NOT RUN")
        add("      To run: `python tests/evaluation/fp_test.py --duration 60 --no-display`")
    add("")
    add("    PERSON/CHAIR SPOT CHECKS: NOT RUN (optional, operator-dependent)")
    add("=" * 74)

    # ---- Limitations ----
    add("  DOCUMENTED LIMITATIONS (carried forward, not tuned away)")
    add("-" * 74)
    add("    1. Class-confusion mislabels (FP-3) — not solved by the area gate.")
    add("    2. Door/stairs are outside YOLO11n COCO classes (FP-4) — unreliable.")
    add("    3. Remaining large-area false positives overlap the genuine cluster.")
    add("    4. couch_visible remains NOT TESTED.")
    add("    5. Single-environment evaluation — not generalizable.")
    add("    6. Relative (not metric) depth — proximity is ordinal.")
    add("    7. Layer 2 correctness is by construction (synthetic ground truth).")
    add("    8. Alert latency excludes TTS/audio playback time.")
    add("=" * 74)
    return "\n".join(L)


def write_report(text: str, timestamp: Optional[str] = None) -> Path:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = _EVAL_DIR / f"phase16_report_{ts}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    return out


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 16 final evaluation orchestrator (offline layers 1+2, "
                    "optional layer 3 hardware capture/analysis)."
    )
    parser.add_argument("--capture-perf", type=str, default=None,
                        help="Copy a completed performance.csv to a timestamped "
                             "phase16_perf_<ts>.csv and analyse it.")
    parser.add_argument("--perf-csv", type=str, default=None, action="append",
                        help="Analyse an existing (already timestamped) perf CSV. "
                             "May be repeated for multiple runs.")
    parser.add_argument("--live-fp-csv", type=str, default=None,
                        help="Path to a live empty-scene FP CSV (from fp_test.py).")
    parser.add_argument("--no-write", action="store_true",
                        help="Print the report but do not write the .txt file.")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Layer 1
    layer1 = layer1_detection()

    # Layer 2
    config = None
    layer2_rows = p16layer2.run_all(config)
    layer2_summary = p16layer2.summarize(layer2_rows)
    if not args.no_write:
        l2_csv = _EVAL_DIR / f"phase16_pipeline_eval_{ts}.csv"
        p16layer2.write_csv(layer2_rows, l2_csv)

    # Layer 3 — perf
    perf_analyses: List[dict] = []
    if args.capture_perf:
        captured = capture_perf_csv(Path(args.capture_perf), timestamp=ts)
        print(f"  Captured perf CSV -> {captured}")
        perf_analyses.append(analyze_perf_csv(captured))
    for pc in (args.perf_csv or []):
        perf_analyses.append(analyze_perf_csv(Path(pc)))

    # Layer 3 — live FP
    live_fp = None
    if args.live_fp_csv:
        p = Path(args.live_fp_csv)
        live_fp = {"available": p.exists(), "source": str(p)}
        if p.exists():
            with open(p, "r", newline="", encoding="utf-8") as f:
                last = None
                for row in csv.DictReader(f):
                    last = row
                if last:
                    live_fp["accepted_per_min"] = last.get("accepted_per_min", "n/a")

    report = build_report(layer1, layer2_rows, layer2_summary, perf_analyses, live_fp)
    print(report)

    if not args.no_write:
        out = write_report(report, timestamp=ts)
        print(f"\n  Phase 16 report written to: {out}")
        if not perf_analyses:
            print("  (Layer 3 hardware: NOT RUN — offline layers 1+2 only.)")
        print("  (No baseline artifacts modified.)")


if __name__ == "__main__":
    main()
