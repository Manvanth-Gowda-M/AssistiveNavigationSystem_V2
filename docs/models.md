# Models — Sources, Sizes, Licenses, and Decisions

Covers README item 21 in depth. All licensing statements here reflect the
actual models used and are cross-referenced with [`licenses.md`](licenses.md).

---

## Models used

| Model | File | Size (on disk) | Runtime | License |
|---|---|---|---|---|
| YOLO11n (detection) | `models/yolo11n.pt` | ~5.4 MB | Ultralytics API on **PyTorch** (CPU) | AGPL-3.0 |
| Depth Anything V2 **Small** (depth) | `models/depth_anything_v2_small.onnx` | ~94.5 MB | **ONNX Runtime** (CPU) | Apache-2.0 |

> The detection model is loaded from `.pt` weights via the Ultralytics API
> (not raw ONNX). The depth model is a float32 ONNX graph run through ONNX
> Runtime. These are two different runtimes in the same process.

---

## Acquisition

- **YOLO11n**: downloaded automatically by Ultralytics on first run (official
  Ultralytics hub). No manual step. `scripts/download_models.py` lists it for
  documentation only.
- **Depth Anything V2 Small**: `python scripts/download_models.py` fetches the
  float32 ONNX from the official HuggingFace source. Verify with
  `python scripts/download_models.py --check`.

Never download model files from unofficial mirrors. After acquisition the
system runs fully offline.

---

## Why the float32 depth model (and not INT8)

An INT8-quantized depth ONNX (`models/depth_anything_v2_small_int8.onnx`,
~26 MB) was evaluated during Phase 6 and **rejected**:

- The INT8 graph uses the **`ConvInteger`** operator, which is **not supported
  by the installed `onnxruntime` (1.23.2)** on this CPU build. Loading/running
  it fails.
- The float32 Small model runs correctly on ONNX Runtime CPU and is the one
  wired into the pipeline.

The INT8 file may still be present on disk from that evaluation; it is **not
used** by the system. [MEASURED — Phase 6 decision]

Measured depth inference cost on the reference CPU (i5-12450H) at the default
`input_size: 252` was on the order of ~110–120 ms per inference, which is why
depth runs every `depth_update_interval` (default 5) frames rather than every
frame. [MEASURED — see `evaluation.md` / config comments]

---

## License constraints (critical)

- **Depth Anything V2 Small = Apache-2.0.** Only the **Small** variant is used.
- **Base / Large / Giant = CC-BY-NC-4.0 (non-commercial).** **Do NOT substitute**
  these variants — doing so changes the project's license posture. This
  constraint is enforced by documentation and the download script's notes.
- **YOLO11n weights** inherit the **AGPL-3.0** Ultralytics license. As an
  open-source research prototype, this is compliant.

Full inventory and legal notes: [`licenses.md`](licenses.md).

---

## COCO note

YOLO11n is trained on the COCO 80-class dataset. Consequences:
- Only COCO classes (and the configured navigation subset) can be detected.
- **`door`, `doorway`, and `stairs` are NOT COCO classes** and are not reliably
  detectable — a documented, safety-critical capability gap. [MEASURED — Phase 4]
- COCO dataset attribution: Lin et al., "Microsoft COCO: Common Objects in
  Context," ECCV 2014 (dataset under CC BY 4.0). Weights only are used; the
  dataset is not redistributed.
