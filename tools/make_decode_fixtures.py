"""Cross-check fixture generator for the browser detection decoder.

Runs the exported ONNX detectors on a deterministic synthetic frame, captures the
raw output tensor exactly (base64 float32, no rounding), and records the boxes
that an independent NumPy postprocess produces.

`tests/js/decodeParity.test.mjs` then runs the browser decoder over the same
bytes and asserts the two implementations agree. That is what makes the JS
decoder trustworthy: it is checked against a second implementation on real model
output, not just against hand-planted tensors.

The synthetic frame is not expected to contain recognisable objects. The
threshold is therefore set very low on purpose, so that a useful number of raw
candidates survive and the box decode, class selection and NMS paths are all
genuinely exercised.

Usage:
    python tools/make_decode_fixtures.py

Requires onnxruntime. On Windows with Store Python, see the note in
scripts/export_web_models.py about installing onnx to a short path.
"""

from __future__ import annotations

import base64
import json
import pathlib

import numpy as np
import onnxruntime as ort

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "tests" / "js" / "fixtures"
MODELS_DIR = REPO / "src" / "assistive_navigation" / "web" / "static" / "models"

SRC_W, SRC_H, SIZE = 640, 480, 320
# Deliberately low: the synthetic frame holds no real objects, and we want the
# decode/NMS paths exercised rather than a semantically meaningful result.
SCORE_TH = 0.0002
IOU_TH = 0.45
MAX_DET = 24


def letterbox_params(sw: int, sh: int, dw: int, dh: int):
    scale = min(dw / sw, dh / sh)
    return scale, (dw - sw * scale) / 2, (dh - sh * scale) / 2


def synthetic_frame():
    """Deterministic RGB frame with hard edges, letterboxed as the client does."""
    rng = np.random.default_rng(7)
    img = np.full((SRC_H, SRC_W, 3), 120, np.uint8)
    img[100:400, 120:260] = 40
    img[200:460, 380:560] = 210
    img[60:140, 500:620] = 90
    noise = rng.integers(0, 24, (SRC_H, SRC_W), dtype=np.int64)
    img[:, :, 1] = (img[:, :, 1] + noise).clip(0, 255)

    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    scale, pad_x, pad_y = letterbox_params(SRC_W, SRC_H, SIZE, SIZE)
    draw_w, draw_h = int(round(SRC_W * scale)), int(round(SRC_H * scale))
    ys = (np.arange(draw_h) / scale).astype(int).clip(0, SRC_H - 1)
    xs = (np.arange(draw_w) / scale).astype(int).clip(0, SRC_W - 1)
    canvas[int(pad_y):int(pad_y) + draw_h, int(pad_x):int(pad_x) + draw_w] = img[ys][:, xs]
    return canvas, (scale, pad_x, pad_y)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def class_aware_nms(boxes, scores, classes, threshold, max_out):
    """Mirrors the greedy, class-aware NMS in vision/decoders.js."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    for i in order:
        if len(keep) >= max_out:
            break
        if any(classes[i] == classes[j] and iou(boxes[i], boxes[j]) > threshold for j in keep):
            continue
        keep.append(int(i))
    return keep


def to_normalised_source(x1, y1, x2, y2, lb):
    scale, pad_x, pad_y = lb

    def convert(value, pad, dimension):
        return float(np.clip(((value - pad) / scale) / dimension, 0.0, 1.0))

    return [
        convert(x1, pad_x, SRC_W),
        convert(y1, pad_y, SRC_H),
        convert(x2, pad_x, SRC_W),
        convert(y2, pad_y, SRC_H),
    ]


def decode_raw(output, lb):
    pred = output[0]
    class_scores = pred[4:]
    best_class = class_scores.argmax(0)
    best_score = class_scores.max(0)

    selected = np.where(best_score >= SCORE_TH)[0]
    cx, cy, w, h = (pred[k, selected] for k in range(4))
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = class_aware_nms(boxes, best_score[selected], best_class[selected], IOU_TH, MAX_DET)
    return [
        {
            "classId": int(best_class[selected][k]),
            "confidence": float(best_score[selected][k]),
            "box": to_normalised_source(*boxes[k], lb),
        }
        for k in keep
    ]


def decode_end_to_end(output, lb):
    results = []
    for row in output[0]:
        if float(row[4]) < SCORE_TH:
            break
        results.append(
            {
                "classId": int(round(float(row[5]))),
                "confidence": float(row[4]),
                "box": to_normalised_source(*row[:4], lb),
            }
        )
        if len(results) >= MAX_DET:
            break
    return results


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas, lb = synthetic_frame()
    tensor = (canvas.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

    fixtures = {
        "generatedBy": "tools/make_decode_fixtures.py",
        "srcWidth": SRC_W,
        "srcHeight": SRC_H,
        "inputSize": SIZE,
        "scoreThreshold": SCORE_TH,
        "iouThreshold": IOU_TH,
        "maxDetections": MAX_DET,
        "models": {},
    }

    for name, family in (("yolo11n_320.onnx", "raw"), ("yolo26n_320.onnx", "e2e")):
        path = MODELS_DIR / name
        if not path.exists():
            print(f"skipping {name}: not exported")
            continue

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        expected = decode_raw(output, lb) if family == "raw" else decode_end_to_end(output, lb)

        fixtures["models"][name] = {
            "family": family,
            "dims": list(output.shape),
            # Exact bytes: rounding the tensor could move a candidate across an
            # NMS boundary and produce a spurious parity failure.
            "dataBase64": base64.b64encode(
                np.ascontiguousarray(output, dtype=np.float32).tobytes()
            ).decode("ascii"),
            "expected": expected,
        }
        print(f"{name}: dims={tuple(output.shape)} reference detections={len(expected)}")

    target = OUT_DIR / "decodeFixtures.json"
    target.write_text(json.dumps(fixtures), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1e6:.2f} MB)")

    if not fixtures["models"]:
        print("no fixtures produced")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

