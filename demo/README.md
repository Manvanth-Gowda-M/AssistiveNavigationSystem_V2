# Phase 18 — Demonstration Evidence Package

This folder is an **evidence container**, not source code. It collects the
auditable artifacts for the final demonstration and validation of the
Assistive Navigation System V2 (research prototype).

Every item is labelled with its evidence type:
**[MEASURED]**, **[SYNTHETIC-DETERMINISTIC]**, **[OPERATOR-LIVE]**,
**[NOT RUN]**, **[DESIGN TARGET]**, **[UNVALIDATED ASSUMPTION]**.

Nothing here is fabricated. Items requiring a physical webcam, live audio, or
a live performance run are marked **[NOT RUN]** and must be produced by an
operator following [`demo_script.md`](demo_script.md).

---

## Contents (generated headlessly)

| File | What it is | Evidence label |
|---|---|---|
| `test_output.txt` | Tail of the full non-hardware regression run | [MEASURED] |
| `git_status.txt` | Working-tree status at capture time | [MEASURED] |
| `git_log.txt` | Recent commit history (Phase chain) | [MEASURED] |
| `protected_file_hashes.txt` | SHA256 + git tree hashes of protected paths | [MEASURED] |
| `project_tree.txt` | Tracked-file listing (git ls-files) | [MEASURED] |
| `demo_script.md` | Operator runbook for the live demonstration | [NOT RUN] until executed |

## Software verification result (this capture)

- **Non-hardware regression: 570 passed, 0 failed** (38 `requires_camera`
  hardware tests deselected). [MEASURED — see `test_output.txt`]
- Protected paths byte-identical to the Phase 17 baseline (config.yaml,
  Phase 4 CSV/report; src/tests/tools/config/data-evaluation git trees).
  [MEASURED — see `protected_file_hashes.txt`]
- HEAD at capture: Phase 18 preparation on top of `9b8a951` (Phase 17).

## Cross-references (authoritative sources — not duplicated here)

- Final report: [`../docs/final_report.md`](../docs/final_report.md)
- Presentation outline: [`../docs/presentation.md`](../docs/presentation.md)
- Evaluation methodology + Phase 4/16 numbers: [`../docs/evaluation.md`](../docs/evaluation.md)
- Phase 16 orchestrator (regenerates the offline report):
  `python tools/phase16_final_eval.py` → writes `data/evaluation/phase16_report_<ts>.txt`
- Reproducibility + checklists: [`../docs/reproducibility.md`](../docs/reproducibility.md)

## Live session performed (2026-09-04)  [OPERATOR-LIVE]

A live webcam session was performed and recorded verbatim in
[`live_session_2026-09-04.md`](live_session_2026-09-04.md). It demonstrated:
startup + all component init, SYSTEM READY, person detection, direction
(ahead/left/right), proximity + escalation, all four alert trigger reasons,
cooldown/duplicate suppression, and clean shutdown. It also recorded honest
caveats: FPS was **below** the Phase 13/14 baseline and **degraded** under CPU
contention (~8 → ~5 → 3.4 FPS), and **track IDs fragmented** at that low frame
rate (stable-ID persistence not cleanly demonstrated). This is a single,
uncontrolled session — **not** a controlled benchmark and **not** an accuracy test.

## Still [NOT RUN] (require separate physical execution — not fabricated)

- Live empty-scene false-positive measurement (`fp_test.py` CSV). **[NOT RUN]**
- Clean performance runs (FPS/latency/CPU/RSS via `--capture-perf`, ≥3 runs with
  other apps closed). **[NOT RUN]**
- Discrete scenarios E (static-object confirmation), F (multiple distinct
  objects), H (brief occlusion), I (difficult/background) as isolated checks. **[NOT RUN]**
- Optional demo video. **[NOT RUN]** (not required)

When an operator completes these, drop the resulting files here (e.g.
`demo/screenshots/`, `demo/live_perf/`, `demo/live_fp/`) and update the labels
from [NOT RUN] to [OPERATOR-LIVE].

---

## Safety / claim boundary

This is a **research prototype**, **not** a certified mobility aid and **not**
a replacement for a trained mobility aid or human assistance. Depth is
**relative, not metric**. Evaluation is **single-environment**. **No**
generalized real-world accuracy is claimed. **Doors and stairs are a known,
unresolved detector limitation** (outside COCO classes).
