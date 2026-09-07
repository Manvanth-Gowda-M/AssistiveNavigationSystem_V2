# On-Device Benchmarking Procedure

How to choose the detector for a given phone. The short version: **run the
benchmark screen on that phone**. Published model benchmarks do not predict
smartphone-browser behaviour, and the difference is routinely large enough to
change the answer.

---

## 1. Why this exists

A model that detects 95% of objects but stalls every few seconds is worse for
this project than a slightly less accurate one that runs continuously and
predictably. The benchmark is therefore weighted towards *predictability*:

- sustained inference rate over 20–30 seconds, not a peak figure;
- p95 latency, not the mean;
- the ratio of p95 to p50, which is what makes guidance feel broken;
- heap growth across the run;
- whether it detected anything at all during a real walk.

A candidate averaging 55 ms with a p95 of 400 ms loses to one averaging 85 ms with
a p95 of 110 ms.

## 2. Candidates

Defined in `js/config/modelRegistry.js`. Exported by
`scripts/export_web_models.py` into `web/static/models/`.

| Candidate | Model | Format | Input | Runtime | Notes |
| --- | --- | --- | --- | --- | --- |
| A | EfficientDet-Lite0 INT8 | TFLite | 320 | MediaPipe Tasks Vision (main thread) | Fetched from Google's model host. GPU delegate where available. |
| A′ | EfficientDet-Lite0 FP16 | TFLite | 320 | MediaPipe | Larger, slightly better. |
| B | YOLO11n | ONNX fp32 + uint8 | 320 | ONNX Runtime Web (worker) | Self-hosted. Raw head, browser-side NMS. |
| C | YOLO26n | ONNX fp32 + uint8 | 320 | ONNX Runtime Web (worker) | Self-hosted. End-to-end head: emits xyxy/score/class directly, so **no browser-side NMS**. |
| fallback | COCO-SSD lite MobileNetV2 | TFJS graph | 300 | TensorFlow.js (main thread) | The detector the pre-upgrade build shipped. Heavier and less accurate; retained because it loads everywhere. |

Output shapes, confirmed from the exported graphs:

- YOLO11n: `[1, 84, 2100]` — rows 0–3 are cx, cy, w, h in input pixels; rows 4–83
  are per-class confidences in 0–1. Needs NMS.
- YOLO26n: `[1, 300, 6]` — x1, y1, x2, y2 in input pixels, score, class id.
  Sorted by descending score. Needs no NMS.

Quantised weights are preferred on the WASM path and fp32 on WebGPU. That
ordering only decides which candidate is *tried first*; the measurement decides
which is used.

## 3. Regenerating the model assets

```bash
python scripts/export_web_models.py                  # both candidates at 320
python scripts/export_web_models.py --imgsz 256      # smaller input
python scripts/export_web_models.py --models yolo11n # one candidate
python scripts/export_web_models.py --no-quantize    # skip uint8
```

Writes `.onnx` files plus `manifest.json` into
`src/assistive_navigation/web/static/models/`. The manifest records byte sizes
and SHA-256 digests; `tests/js/integrity.test.mjs` checks the deployed files
against it, and `modelCache.js` validates the cached bytes against the size
declared in the registry.

After re-exporting, update `approxBytes` in `modelRegistry.js` (the integrity test
will tell you if you forget) and regenerate the decoder parity fixtures:

```bash
python tools/make_decode_fixtures.py
npm test
```

On Windows with the Microsoft Store Python distribution, the `onnx` wheel can
fail to install into `site-packages` because of MAX_PATH limits. Install it into a
short path instead:

```
python -m pip install --target C:\op onnx
set PYTHONPATH=C:\op
```

## 4. Running the benchmark on a phone

1. Serve the client over HTTPS or `localhost` (the camera requires a secure
   context) and open it on the device.
2. Wait for **Vision ready**.
3. Open *Developer tools → Benchmark*.
4. Tick the candidates to compare. By default one representative per candidate
   letter is selected.
5. Press **Run benchmark** and follow the on-screen instruction for each phase.

Each candidate runs through six operator-guided conditions, roughly 27 seconds
total per candidate:

| Condition | Duration | What it exposes |
| --- | --- | --- |
| Hold the phone still | 5 s | Best-case latency and detection stability. |
| Walk slowly | 5 s | Mild motion blur; tracking continuity. |
| Walk at normal pace | 5 s | The real operating condition. |
| Pan quickly left and right | 4 s | Motion-blur tolerance and scene-change handling. |
| Point at an indoor scene | 4 s | Cluttered, low-light, many small objects. |
| Point at an outdoor scene | 4 s | High dynamic range, distant objects, vehicles. |

Every sample is tagged with the active condition, so the results show *where* a
candidate falls apart rather than an average that hides it. The per-condition
breakdown is in the JSON block under the table.

These conditions cannot be automated on a phone — "walk normally" is a human
action — which is why the screen prompts an operator rather than pretending to
run itself.

## 5. Reading the results

Columns are exactly those required for a decision:

`MODEL · BACKEND · INPUT · INF FPS · AVG MS · P95 MS · HEAP MB · DET/FRAME · DROPPED · CAM FPS · VERDICT`

Scoring (`verdictFor()` in `js/ui/benchmarkScreen.js`), out of 100:

| Component | Weight | Full marks at |
| --- | --- | --- |
| Sustained inference rate | 40 | ≥ 12 FPS |
| p95 latency | 30 | ≤ 90 ms |
| Predictability (p95 / p50) | 20 | ≤ 1.8× |
| Heap growth over the run | 10 | ≤ 40 MB |
| Detections per frame | 10 | ≥ 0.4 |

Grades: EXCELLENT ≥ 85, GOOD ≥ 65, MARGINAL ≥ 45, POOR below that. A candidate is
FAILED outright if it produced fewer than 20 samples or failed more than 5% of
inferences.

`HEAP MB` is blank on browsers that do not expose `performance.memory` (Safari,
Firefox). That is reported as unavailable rather than treated as zero growth.

## 6. Choosing

Pick the highest-scoring candidate that reaches at least GOOD. If nothing does:

1. Switch the quality mode to **Fast** and re-run. That drops the input size one
   step and disables the refinement stage.
2. If still nothing reaches GOOD, the device belongs on the LOW tier; expect
   8 FPS operation and no refinement stage.
3. If the ONNX candidates all fail but MediaPipe or TensorFlow.js passes, the
   device has a broken WASM or WebGPU path. The recovery ladder will land there
   on its own, but recording it saves a support conversation.

Record the outcome per device. `docs/field_testing.md` has a table to fill in.

## 7. Device coverage

Benchmark at minimum one low-end Android, one mid-range Android and one high-end
Android; add an iPhone where available. The point is not to find the fastest
device but to confirm the **degradation** works: a LOW-tier device should end up
at a smaller input, a lower rate and no refinement stage, and should still walk.

Tier classification is in `classifyDeviceTier()`. It is deliberately pessimistic —
a device that reports nothing useful lands in MID, not HIGH, because guessing high
and thermally collapsing mid-demo is the worse failure.

## 8. Sustained-load behaviour

The benchmark's 27 seconds per candidate is long enough to expose startup cost and
latency spikes but **not** long enough to expose thermal throttling. For that,
follow the 5- and 10-minute walking procedures in `docs/field_testing.md` and
watch the diagnostics panel:

- `Sustained load` flips from `stable` to `throttled` when p95 latency degrades by
  more than 1.5× relative to the session's best measurement.
- `Target FPS` should step *down* under throttling and the ceiling should drop
  with it, so the ramp-up logic cannot immediately undo the correction.
- `Refinement stage` should switch itself off before the rate falls far.

Sustained performance is the goal, not fast for thirty seconds.

## 9. Automated checks that back this up

`npm test` covers the parts that do not need a phone:

- `decodeParity.test.mjs` runs the browser decoder over real ONNX output from both
  exported models and compares it against an independent NumPy postprocess.
- `system.test.mjs` covers the backend plan, warm-up gate, tensor sanity check,
  thread gating and profile bounds.
- `pipeline.test.mjs` covers the scheduler's no-queue guarantee and its rate
  controller, including the thermal-regression path.

These cannot tell you which model is fastest on a given phone. They can tell you
that the machinery which decides is correct.
