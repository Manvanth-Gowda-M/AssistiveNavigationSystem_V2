"""
Phase 15 — False-Positive Area Analysis (OFFLINE, READ-ONLY)
=============================================================
Phase: 15
Status: IMPLEMENTED

PURPOSE
-------
Decide — with evidence, not assumption — whether a calibrated
`temporal.min_area_norm` threshold can suppress the dominant Phase 4
false positives WITHOUT unacceptable loss of genuine navigation objects.

This tool NEVER modifies phase4_results.csv or phase4_report.txt.
It only READS them. All output goes to NEW, separate files.

WHAT IT DOES
------------
1. Parses data/evaluation/phase4_results.csv (read-only).
2. For every ACCEPTED detection row, recomputes:
       area_norm = (x2 - x1) * (y2 - y1) / (FRAME_W * FRAME_H)
   using the stored 640x480 bounding boxes.
3. Splits accepted detections into two evidence groups:
       GENUINE : accepted rows whose detected_class matches the scenario
                 ground truth, drawn from true-positive / multi-object
                 scenarios (real navigation objects that were present).
       FALSE_POSITIVE : accepted rows from scenes the user declared to
                 contain NO navigation object (empty scene, empty desk,
                 usb charger, pen/pencil probes).
4. Reports the area_norm distribution (min / p10 / median / mean / p90 /
   max) for each group.
5. Sweeps candidate min_area_norm thresholds and, for each, computes:
       - FP accepted SUPPRESSED vs RETAINED  (higher suppression = better)
       - GENUINE accepted RETAINED vs LOST   (higher retention = better)
   broken down per scenario and per class.
6. Writes a timestamped analysis report + machine-readable CSV to
   data/evaluation/phase15_analysis_<timestamp>.{txt,csv}

WHAT IT DOES NOT DO
-------------------
- Does NOT change any threshold.
- Does NOT touch confidence_threshold (stays 0.35 by policy).
- Does NOT run the detector, tracker, depth, or TTS.
- Does NOT overwrite the Phase 4 baseline.
- Does NOT make the final decision for you — it prints the evidence and a
  recommendation; a human approves the threshold.

IMPORTANT MEASUREMENT CAVEAT
----------------------------
Phase 4 "accepted" reflects DetectionFilter output ONLY
(confidence >= 0.35 + navigation class list). No tracking and no temporal
confirmation were applied when the CSV was recorded. Therefore this analysis
measures the PRE-TEMPORAL area separability of genuine vs false-positive
detections. The temporal confirmation gate (confirmation_frames) still
applies at runtime and is complementary to the area gate.

Usage:
    .venv/Scripts/python.exe tools/phase15_fp_analysis.py
    .venv/Scripts/python.exe tools/phase15_fp_analysis.py --no-write
"""

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CSV_PATH = _PROJECT_ROOT / "data" / "evaluation" / "phase4_results.csv"
_OUT_DIR = _PROJECT_ROOT / "data" / "evaluation"

# The Phase 4 evaluation ran the camera at 640x480 (config camera.width/height).
FRAME_W = 640
FRAME_H = 480
FRAME_AREA = float(FRAME_W * FRAME_H)

# Scenarios whose declared ground truth is "no navigation object present".
# Accepted detections in these scenes are observed false positives.
FP_SCENARIOS = {
    "empty_scene_fp_rate",
    "empty_desk",
    "usb_charger",
    "pen_or_pencil",
}

# Candidate thresholds to sweep. 0.02 is the current default (baseline).
# The rest span the plausible range between the FP cluster and genuine cluster.
CANDIDATE_THRESHOLDS = [
    0.02,  # current default (baseline)
    0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
]


# ---------------------------------------------------------------------------
# Row model
# ---------------------------------------------------------------------------

class AcceptedRow:
    """A single ACCEPTED detection row with recomputed area_norm."""

    __slots__ = (
        "scenario", "category", "ground_truth", "detected_class",
        "confidence", "area_norm", "is_genuine", "is_false_positive",
    )

    def __init__(
        self,
        scenario: str,
        category: str,
        ground_truth: str,
        detected_class: str,
        confidence: float,
        area_norm: float,
        is_genuine: bool,
        is_false_positive: bool,
    ) -> None:
        self.scenario = scenario
        self.category = category
        self.ground_truth = ground_truth
        self.detected_class = detected_class
        self.confidence = confidence
        self.area_norm = area_norm
        self.is_genuine = is_genuine
        self.is_false_positive = is_false_positive


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _safe_float(value: str) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def compute_area_norm(x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Recompute normalised bbox area from stored 640x480 coordinates.
    area_norm = bbox_pixel_area / frame_pixel_area, clamped to [0, 1].
    """
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = (w * h) / FRAME_AREA
    if area < 0.0:
        return 0.0
    if area > 1.0:
        return 1.0
    return area


def load_accepted_rows(csv_path: Path) -> List[AcceptedRow]:
    """
    Read the Phase 4 CSV (read-only) and return only ACCEPTED detections,
    each annotated with a recomputed area_norm and a genuine/FP label.

    Classification rules (evidence-based, matching phase4_eval semantics):
      - is_false_positive: scenario is in FP_SCENARIOS (declared empty scenes)
      - is_genuine: NOT an FP scenario AND detected_class == ground_truth
                    (a real navigation object that was actually present)
      - Rows that are accepted but are wrong-class mislabels in a TP scene
        (e.g. table -> bottle) are counted as NEITHER genuine nor FP here;
        they are reported separately as "mislabel" so they do not distort
        the two clean evidence distributions. This is FP-3 in the design,
        which the area gate is not expected to solve.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 4 CSV not found: {csv_path}")

    rows: List[AcceptedRow] = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if str(r.get("accepted", "")).strip().lower() != "true":
                continue

            x1 = _safe_float(r.get("bbox_x1"))
            y1 = _safe_float(r.get("bbox_y1"))
            x2 = _safe_float(r.get("bbox_x2"))
            y2 = _safe_float(r.get("bbox_y2"))
            if None in (x1, y1, x2, y2):
                # Accepted row without usable bbox — cannot compute area; skip
                continue

            area_norm = compute_area_norm(x1, y1, x2, y2)

            scenario = (r.get("scenario") or "").strip()
            category = (r.get("category") or "").strip()
            ground_truth = (r.get("ground_truth") or "").strip()
            detected_class = (r.get("detected_class") or "").strip()
            confidence = _safe_float(r.get("confidence")) or 0.0

            is_fp = scenario in FP_SCENARIOS
            is_genuine = (not is_fp) and (detected_class == ground_truth)

            rows.append(AcceptedRow(
                scenario=scenario,
                category=category,
                ground_truth=ground_truth,
                detected_class=detected_class,
                confidence=confidence,
                area_norm=area_norm,
                is_genuine=is_genuine,
                is_false_positive=is_fp,
            ))

    return rows


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linear-interpolation percentile. pct in [0,100]. Assumes sorted input."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = k - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def describe(vals: List[float]) -> Dict[str, float]:
    """Return distribution summary for a list of area_norm values."""
    if not vals:
        return {
            "n": 0, "min": float("nan"), "p10": float("nan"),
            "median": float("nan"), "mean": float("nan"),
            "p90": float("nan"), "max": float("nan"),
        }
    s = sorted(vals)
    return {
        "n": len(s),
        "min": s[0],
        "p10": _percentile(s, 10),
        "median": _percentile(s, 50),
        "mean": statistics.mean(s),
        "p90": _percentile(s, 90),
        "max": s[-1],
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def sweep_thresholds(
    rows: List[AcceptedRow],
    thresholds: List[float],
) -> List[dict]:
    """
    For each candidate threshold, compute FP suppression and genuine retention.

    A detection is RETAINED if area_norm >= threshold (i.e. it would pass the
    area gate). A detection is SUPPRESSED/LOST if area_norm < threshold.

    Returns a list of dicts, one per threshold.
    """
    genuine = [r for r in rows if r.is_genuine]
    fps = [r for r in rows if r.is_false_positive]

    results = []
    for t in thresholds:
        fp_suppressed = sum(1 for r in fps if r.area_norm < t)
        fp_retained = len(fps) - fp_suppressed
        gen_retained = sum(1 for r in genuine if r.area_norm >= t)
        gen_lost = len(genuine) - gen_retained

        results.append({
            "threshold": t,
            "fp_total": len(fps),
            "fp_suppressed": fp_suppressed,
            "fp_retained": fp_retained,
            "fp_suppression_pct": (100.0 * fp_suppressed / len(fps)) if fps else 0.0,
            "gen_total": len(genuine),
            "gen_retained": gen_retained,
            "gen_lost": gen_lost,
            "gen_retention_pct": (100.0 * gen_retained / len(genuine)) if genuine else 0.0,
        })
    return results


def per_scenario_breakdown(
    rows: List[AcceptedRow],
    threshold: float,
) -> Dict[str, dict]:
    """
    For a specific threshold, break down retained/suppressed per scenario.
    """
    by_scenario: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "retained": 0, "suppressed": 0,
                 "is_fp": False, "is_genuine": False}
    )
    for r in rows:
        # Only consider rows that are clearly genuine or clearly FP
        if not (r.is_genuine or r.is_false_positive):
            continue
        b = by_scenario[r.scenario]
        b["total"] += 1
        if r.area_norm >= threshold:
            b["retained"] += 1
        else:
            b["suppressed"] += 1
        b["is_fp"] = r.is_false_positive
        b["is_genuine"] = r.is_genuine
    return dict(by_scenario)


def per_class_genuine_retention(
    rows: List[AcceptedRow],
    threshold: float,
) -> Dict[str, dict]:
    """Genuine-object retention per detected class at a given threshold."""
    by_class: Dict[str, dict] = defaultdict(
        lambda: {"total": 0, "retained": 0, "lost": 0}
    )
    for r in rows:
        if not r.is_genuine:
            continue
        b = by_class[r.detected_class]
        b["total"] += 1
        if r.area_norm >= threshold:
            b["retained"] += 1
        else:
            b["lost"] += 1
    return dict(by_class)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def recommend(sweep: List[dict], current_default: float = 0.02) -> dict:
    """
    Recommend a threshold using an evidence-based rule:

      - Require ZERO genuine loss (gen_retention_pct == 100).
      - Among zero-loss candidates, pick the one with the highest FP
        suppression.
      - If several tie on suppression, pick the smallest threshold (least
        aggressive / most conservative that still achieves it).
      - If NO candidate above the current default improves FP suppression
        without genuine loss, recommend NO CHANGE and flag insufficiency.

    Returns a dict with the recommendation and the rationale.
    """
    zero_loss = [s for s in sweep if s["gen_lost"] == 0]

    baseline = next((s for s in sweep if abs(s["threshold"] - current_default) < 1e-9), None)
    baseline_fp_supp = baseline["fp_suppressed"] if baseline else 0

    if not zero_loss:
        return {
            "action": "NO_CHANGE",
            "threshold": current_default,
            "reason": "No candidate threshold achieves zero genuine-object loss.",
            "sufficient": False,
        }

    # Best zero-loss candidate by FP suppression, then smallest threshold
    zero_loss_sorted = sorted(
        zero_loss,
        key=lambda s: (-s["fp_suppressed"], s["threshold"]),
    )
    best = zero_loss_sorted[0]

    if best["fp_suppressed"] <= baseline_fp_supp:
        return {
            "action": "NO_CHANGE",
            "threshold": current_default,
            "reason": (
                "No zero-genuine-loss threshold improves FP suppression beyond "
                f"the current default ({current_default}). Area gate alone is "
                "not sufficient; do NOT force a change."
            ),
            "sufficient": False,
        }

    return {
        "action": "CHANGE",
        "threshold": best["threshold"],
        "reason": (
            f"Threshold {best['threshold']} suppresses "
            f"{best['fp_suppressed']}/{best['fp_total']} FP "
            f"({best['fp_suppression_pct']:.1f}%) with "
            f"{best['gen_retention_pct']:.1f}% genuine retention "
            f"(0 genuine lost)."
        ),
        "sufficient": True,
        "best": best,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(
    rows: List[AcceptedRow],
    sweep: List[dict],
    recommendation: dict,
) -> str:
    genuine = [r for r in rows if r.is_genuine]
    fps = [r for r in rows if r.is_false_positive]
    mislabel = [r for r in rows if not (r.is_genuine or r.is_false_positive)]

    gd = describe([r.area_norm for r in genuine])
    fd = describe([r.area_norm for r in fps])

    L: List[str] = []
    add = L.append

    add("=" * 72)
    add("  PHASE 15 — FALSE-POSITIVE AREA ANALYSIS (OFFLINE, READ-ONLY)")
    add("=" * 72)
    add(f"    Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"    Source CSV : data/evaluation/phase4_results.csv (READ-ONLY)")
    add(f"    Frame size : {FRAME_W}x{FRAME_H}  (area_norm recomputed from bbox)")
    add("")
    add("    MEASUREMENT CAVEAT: Phase 4 'accepted' = DetectionFilter output")
    add("    ONLY (confidence>=0.35 + class list). No tracking / temporal")
    add("    confirmation was applied. This measures PRE-TEMPORAL area")
    add("    separability. The runtime temporal gate is complementary.")
    add("=" * 72)
    add("  EVIDENCE GROUPS (accepted detections only)")
    add("-" * 72)
    add(f"    GENUINE (real object present, class matches GT) : {len(genuine)}")
    add(f"    FALSE POSITIVE (declared-empty scenes)          : {len(fps)}")
    add(f"    MISLABEL (wrong class in TP scene, e.g. FP-3)   : {len(mislabel)}")
    add("      (mislabels excluded from the two clean distributions;")
    add("       the area gate is not expected to solve class confusion)")
    add("=" * 72)
    add("  AREA_NORM DISTRIBUTIONS")
    add("-" * 72)
    add("    Group      n     min    p10   median   mean    p90    max")
    add("    -----------------------------------------------------------------")
    add(f"    GENUINE  {gd['n']:>4}  {gd['min']:.4f} {gd['p10']:.4f} "
        f"{gd['median']:.4f} {gd['mean']:.4f} {gd['p90']:.4f} {gd['max']:.4f}")
    add(f"    FALSE_P  {fd['n']:>4}  {fd['min']:.4f} {fd['p10']:.4f} "
        f"{fd['median']:.4f} {fd['mean']:.4f} {fd['p90']:.4f} {fd['max']:.4f}")
    add("")
    # Overlap diagnosis
    if genuine and fps:
        overlap_lo = max(gd["min"], fd["min"])
        overlap_hi = min(gd["max"], fd["max"])
        if overlap_hi > overlap_lo:
            add(f"    OVERLAP RANGE (area_norm): [{overlap_lo:.4f}, {overlap_hi:.4f}]")
            add("    Genuine and FP area ranges OVERLAP in this band — a clean")
            add("    single-threshold separation is not guaranteed. See sweep.")
        else:
            add("    NO OVERLAP: genuine and FP area ranges are fully separable.")
    add("=" * 72)
    add("  THRESHOLD SWEEP  (RETAINED = area_norm >= threshold)")
    add("-" * 72)
    add("    thresh   FP_supp/tot  FP_supp%   GEN_ret/tot  GEN_ret%   GEN_lost")
    add("    -----------------------------------------------------------------")
    for s in sweep:
        marker = "  <= current" if abs(s["threshold"] - 0.02) < 1e-9 else ""
        add(f"    {s['threshold']:.2f}    "
            f"{s['fp_suppressed']:>4}/{s['fp_total']:<4}   "
            f"{s['fp_suppression_pct']:>5.1f}%    "
            f"{s['gen_retained']:>4}/{s['gen_total']:<4}   "
            f"{s['gen_retention_pct']:>5.1f}%    "
            f"{s['gen_lost']:>4}{marker}")
    add("=" * 72)
    add("  RECOMMENDATION")
    add("-" * 72)
    add(f"    ACTION     : {recommendation['action']}")
    add(f"    THRESHOLD  : {recommendation['threshold']}")
    add(f"    SUFFICIENT : {recommendation['sufficient']}")
    add(f"    REASON     : {recommendation['reason']}")
    add("")
    if not recommendation["sufficient"]:
        add("    >>> Per the Phase 15 design + approval condition #15, the area")
        add("    >>> gate is reported as INSUFFICIENT. Do NOT force a threshold")
        add("    >>> change and do NOT add an unapproved filtering mechanism.")
    add("=" * 72)

    # Per-scenario + per-class breakdown at the recommended (or default) threshold
    t = recommendation["threshold"]
    add(f"  PER-SCENARIO BREAKDOWN @ threshold={t}")
    add("-" * 72)
    scn = per_scenario_breakdown(rows, t)
    add("    scenario                     type      total  retained  suppressed")
    add("    -----------------------------------------------------------------")
    for name in sorted(scn.keys()):
        b = scn[name]
        typ = "FP" if b["is_fp"] else ("GENUINE" if b["is_genuine"] else "-")
        add(f"    {name:<28} {typ:<8} {b['total']:>5}  {b['retained']:>8}  "
            f"{b['suppressed']:>10}")
    add("")
    add(f"  PER-CLASS GENUINE RETENTION @ threshold={t}")
    add("-" * 72)
    pcls = per_class_genuine_retention(rows, t)
    add("    class            total  retained  lost")
    add("    ---------------------------------------")
    for name in sorted(pcls.keys()):
        b = pcls[name]
        add(f"    {name:<16} {b['total']:>5}  {b['retained']:>8}  {b['lost']:>4}")
    add("=" * 72)
    add("  LIMITATIONS")
    add("-" * 72)
    add("    1. Pre-temporal measurement (no tracking/confirmation applied).")
    add("    2. Single room/lighting/camera — not a generalisable benchmark.")
    add("    3. Mislabels (FP-3) and door/stair COCO gap (FP-4) are NOT")
    add("       addressed by the area gate and remain documented limitations.")
    add("    4. couch_visible remains NOT TESTED.")
    add("=" * 72)
    return "\n".join(L)


def write_outputs(report_text: str, sweep: List[dict]) -> Dict[str, Path]:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = _OUT_DIR / f"phase15_analysis_{ts}.txt"
    csv_path = _OUT_DIR / f"phase15_analysis_{ts}.csv"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold", "fp_total", "fp_suppressed", "fp_retained",
            "fp_suppression_pct", "gen_total", "gen_retained", "gen_lost",
            "gen_retention_pct",
        ])
        for s in sweep:
            w.writerow([
                s["threshold"], s["fp_total"], s["fp_suppressed"],
                s["fp_retained"], round(s["fp_suppression_pct"], 2),
                s["gen_total"], s["gen_retained"], s["gen_lost"],
                round(s["gen_retention_pct"], 2),
            ])
    return {"txt": txt_path, "csv": csv_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_analysis(csv_path: Path = _CSV_PATH) -> dict:
    """
    Run the full analysis and return a structured result dict.
    Importable for tests (does not write files).
    """
    rows = load_accepted_rows(csv_path)
    sweep = sweep_thresholds(rows, CANDIDATE_THRESHOLDS)
    recommendation = recommend(sweep, current_default=0.02)
    return {"rows": rows, "sweep": sweep, "recommendation": recommendation}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 15 offline false-positive area analysis (read-only)."
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Print the report but do not write output files."
    )
    args = parser.parse_args()

    result = run_analysis()
    report = build_report(
        result["rows"], result["sweep"], result["recommendation"]
    )
    print(report)

    if not args.no_write:
        paths = write_outputs(report, result["sweep"])
        print(f"\n  Analysis report : {paths['txt']}")
        print(f"  Analysis CSV    : {paths['csv']}")
        print("\n  (phase4_results.csv and phase4_report.txt were NOT modified.)")


if __name__ == "__main__":
    main()
