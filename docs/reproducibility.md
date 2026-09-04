# Reproducibility & Checklists

Covers README item 28 (reproducing Phase 4/15/16 results) and provides the
reproducibility, clean-machine, final-evaluation, and final-demo checklists.

Every result below carries its evidence label:
**[MEASURED]**, **[SYNTHETIC-DETERMINISTIC]**, **[OPERATOR-LIVE / NOT RUN]**.

---

## 28. Reproducing Phase 4 / 15 / 16 results

### Phase 4 — physical detection evaluation  [MEASURED, operator-run]
The Phase 4 results were produced by an **interactive, physical** evaluation
with a live camera and manually arranged scenes. The authoritative outputs are
committed and **must not be overwritten**:
- `data/evaluation/phase4_results.csv`
- `data/evaluation/phase4_report.txt`

To re-run the physical evaluation (produces NEW output; the harness writes to
its own files — do not overwrite the baselines):
```powershell
python tests/evaluation/phase4_eval.py --list-scenarios
python tests/evaluation/phase4_eval.py            # interactive; needs a camera + operator
```
The pure evaluation logic (no camera) is covered by
`tests/unit/test_phase4_logic.py`.

### Phase 15 — false-positive reduction  [MEASURED, offline]
Phase 15 is fully reproducible offline from the Phase 4 CSV (read-only):
```powershell
.venv\Scripts\python.exe tools/phase15_fp_analysis.py
.venv\Scripts\python.exe -m pytest tests/evaluation/phase15_fp_replay.py -q
```
Expected (reproduced from the raw data): the `min_area_norm` 0.02 → 0.15 change
suppresses the dominant empty-scene false positives while retaining all measured
genuine objects. Exact before/after numbers are in [`evaluation.md`](evaluation.md).

### Phase 16 — three-layer evaluation
```powershell
# Layer 1 (offline, reproduces Phase 15 from phase4_results.csv) + Layer 2:
.venv\Scripts\python.exe tools/phase16_final_eval.py

# Layer 2 alone (synthetic, deterministic):
.venv\Scripts\python.exe tests/evaluation/phase16_pipeline_eval.py

# Deterministic assertions:
.venv\Scripts\python.exe -m pytest tests/unit/test_phase16.py -q
```
- Layer 1 = [MEASURED] (from Phase 4 data).
- Layer 2 = [SYNTHETIC-DETERMINISTIC] (correct by construction; pipeline logic,
  not real-world accuracy).
- Layer 3 = [OPERATOR-LIVE / NOT RUN] unless you run the hardware steps in
  [`usage.md`](usage.md) §27.

---

## Reproducibility checklist

- [ ] Repo obtained; Python 3.9–3.11 available (tested 3.10.11).
- [ ] `.venv` created and activated.
- [ ] `pip install -r requirements.txt` (exact pinned versions).
- [ ] `python scripts/download_models.py --check`, then download the depth model.
- [ ] Full non-hardware regression passes: `pytest ... -m "not requires_camera"` → **570 passed** [MEASURED].
- [ ] `tools/phase15_fp_analysis.py` + `phase15_fp_replay.py` reproduce Phase 15.
- [ ] `tools/phase16_final_eval.py` reproduces Layer 1 numbers and Layer 2 = all checks pass.
- [ ] `tests/evaluation/phase16_pipeline_eval.py` = all checks pass [SYNTHETIC-DETERMINISTIC].
- [ ] Baseline files unchanged (SHA256): `phase4_results.csv`, `phase4_report.txt`, `config.yaml`.
- [ ] (Optional) Hardware runs captured per [`usage.md`](usage.md) §27 [OPERATOR-LIVE].

---

## Clean-machine setup checklist

- [ ] OS is Windows 10/11 (primary) or a supported Linux/macOS.
- [ ] Python 3.9–3.11 installed and on PATH.
- [ ] Working webcam and audio output present.
- [ ] `.venv` created; dependencies installed from `requirements.txt`.
- [ ] Internet available for the **first** run (model auto-download); offline after.
- [ ] Depth model downloaded via `scripts/download_models.py`.
- [ ] `python -m assistive_navigation` starts and shows the status banner.
- [ ] `--headless` mode also starts cleanly.

---

## Final-evaluation checklist

- [ ] Non-hardware regression: **570 passed, 0 failed** [MEASURED].
- [ ] Phase 15 replay reproduces the documented before/after.
- [ ] Phase 16 Layer 1 reproduces Phase 15 numbers from the Phase 4 CSV.
- [ ] Phase 16 Layer 2: direction (≥21 positions + boundaries), proximity bands
      + monotonicity, temporal confirmation, area gate @0.15, tracking,
      alert correctness/dedup/selection/latency, multi/partial/difficult — all pass.
- [ ] Alert latency reported as pipeline-internal, **excluding** TTS/audio.
- [ ] (Optional) ≥3 hardware performance runs captured, FPS compared to the
      Phase 13–14 baseline range; single low run not treated as failure.
- [ ] Live empty-scene FP run recorded as a measurement (if performed).
- [ ] Anything not executed is explicitly marked **NOT RUN**.
- [ ] No precision/recall/mAP/accuracy/generalization claimed anywhere.

---

## Final-demo checklist

- [ ] Safe indoor environment; a sighted person present.
- [ ] Safety disclaimer stated to observers (research prototype, not a mobility aid).
- [ ] Camera and audio confirmed working before starting.
- [ ] Demonstrate: person ahead → spoken alert; object left/right → correct direction.
- [ ] Show a debug-window run and a `--headless` audio-only run.
- [ ] Show the honest limitations live: a class-confusion example and the
      door/stairs gap (do NOT rely on it near stairs).
- [ ] Have `evaluation.md` and this checklist available for reviewer questions.
- [ ] Do not present synthetic Layer 2 results as real-world accuracy.

---

## Verifying baselines are unchanged

The three key baseline files should hash-match the committed versions. Compute:
```powershell
Get-FileHash config\config.yaml, data\evaluation\phase4_results.csv, data\evaluation\phase4_report.txt -Algorithm SHA256
```
Compare against the values recorded in the Phase 17 commit message / CHANGELOG.
Any difference means a protected file was altered and must be restored.
