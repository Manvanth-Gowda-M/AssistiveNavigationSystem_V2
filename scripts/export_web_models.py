"""
Export browser-ready detector assets for the real-time web navigation client.
=============================================================================

Produces self-hosted ONNX detectors under
``src/assistive_navigation/web/static/models/`` so that:

* the browser client has no runtime dependency on a third-party model CDN,
* the assets are served from the same origin as the app (cacheable in
  IndexedDB by ``js/vision/modelCache.js``),
* the GitHub Pages deployment ships the models with the site.

Candidates produced
-------------------
====================  ==================================================
Candidate B           YOLO11n exported to ONNX at the configured input
                      size, plus a dynamically quantised uint8 variant
                      for the WASM/CPU path.
Candidate C           YOLO26n, if the weights are resolvable via
                      ultralytics. Skipped (not failed) when the model
                      family is unavailable in the installed version.
====================  ==================================================

Candidate A (EfficientDet-Lite0 INT8) is a MediaPipe ``.tflite`` asset that is
downloaded and cached by the client at runtime; it is not produced here.

Usage
-----
    python scripts/export_web_models.py
    python scripts/export_web_models.py --imgsz 320 --no-quantize
    python scripts/export_web_models.py --models yolo11n

Requires ``ultralytics``, ``torch`` and ``onnx``. On Windows with the Microsoft
Store Python distribution the ``onnx`` wheel may fail to install into
``site-packages`` because of MAX_PATH limits; install it into a short target
directory instead and point ``PYTHONPATH`` at it::

    python -m pip install --target C:\\op onnx
    set PYTHONPATH=C:\\op
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("export_web_models")

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = REPO_ROOT / "models"
STATIC_MODELS_DIR = REPO_ROOT / "src" / "assistive_navigation" / "web" / "static" / "models"

# Classes the walking-guidance pipeline actually consumes. Recorded in the
# manifest so the browser client can filter decode output cheaply instead of
# post-processing all 80 COCO classes.
NAVIGATION_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "bench",
    "dog",
    "backpack",
    "handbag",
    "suitcase",
    "sports ball",
    "bottle",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "refrigerator",
    "vase",
]


@dataclass
class ExportTarget:
    """One detector candidate to export."""

    key: str
    weights: str
    imgsz: int
    label: str
    notes: str = ""
    quantize: bool = True
    produced: List[Dict[str, Any]] = field(default_factory=list)
    skipped_reason: Optional[str] = None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_weights(name: str) -> Optional[Path]:
    """Prefer a checkpoint already vendored in ``models/`` over a download."""
    local = WEIGHTS_DIR / name
    if local.exists():
        LOGGER.info("Using vendored weights: %s", local)
        return local
    LOGGER.info("No vendored weights at %s; ultralytics will resolve %s", local, name)
    return None


def export_onnx(target: ExportTarget, opset: int) -> Optional[Path]:
    """Export ultralytics weights to ONNX. Returns the produced .onnx path."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"ultralytics unavailable: {exc}") from exc

    weights_path = resolve_weights(target.weights)
    load_arg = str(weights_path) if weights_path else target.weights

    LOGGER.info("Loading %s", load_arg)
    model = YOLO(load_arg)

    LOGGER.info(
        "Exporting %s to ONNX (imgsz=%d, opset=%d, static batch 1)",
        target.key,
        target.imgsz,
        opset,
    )
    exported = model.export(
        format="onnx",
        imgsz=target.imgsz,
        opset=opset,
        dynamic=False,
        simplify=False,
        half=False,
        nms=False,
        device="cpu",
    )
    exported_path = Path(exported)
    if not exported_path.exists():
        raise RuntimeError(f"ultralytics reported {exported} but the file is missing")
    return exported_path


def quantize_uint8(src: Path, dst: Path) -> bool:
    """Dynamic uint8 quantisation for the WASM/CPU execution path."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Skipping quantisation (onnxruntime.quantization unavailable: %s)", exc)
        return False

    preprocessed = dst.with_name(dst.stem + "_pre.onnx")
    try:
        try:
            quant_pre_process(str(src), str(preprocessed), skip_symbolic_shape=False)
            quant_src = preprocessed
        except Exception as exc:
            LOGGER.warning("quant_pre_process failed (%s); quantising the raw graph", exc)
            quant_src = src

        quantize_dynamic(
            model_input=str(quant_src),
            model_output=str(dst),
            weight_type=QuantType.QUInt8,
            per_channel=False,
            reduce_range=False,
        )
        return dst.exists()
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Dynamic quantisation failed for %s: %s", src.name, exc)
        return False
    finally:
        if preprocessed.exists():
            preprocessed.unlink(missing_ok=True)


def describe_onnx(path: Path) -> Dict[str, Any]:
    """Read input/output tensor metadata straight from the graph."""
    info: Dict[str, Any] = {}
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        info["inputName"] = sess.get_inputs()[0].name
        info["inputShape"] = [d if isinstance(d, int) else -1 for d in sess.get_inputs()[0].shape]
        info["outputName"] = sess.get_outputs()[0].name
        info["outputShape"] = [d if isinstance(d, int) else -1 for d in sess.get_outputs()[0].shape]
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("Could not introspect %s: %s", path.name, exc)
    return info


def process_target(target: ExportTarget, opset: int, do_quantize: bool) -> ExportTarget:
    STATIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        exported = export_onnx(target, opset)
    except Exception as exc:
        LOGGER.warning("Candidate %s skipped: %s", target.key, exc)
        target.skipped_reason = str(exc)
        return target

    fp32_name = f"{target.key}_{target.imgsz}.onnx"
    fp32_path = STATIC_MODELS_DIR / fp32_name
    shutil.copy2(exported, fp32_path)
    LOGGER.info("Wrote %s (%.2f MB)", fp32_path.name, fp32_path.stat().st_size / 1e6)

    meta = describe_onnx(fp32_path)
    target.produced.append(
        {
            "file": fp32_name,
            "quantization": "fp32",
            "bytes": fp32_path.stat().st_size,
            "sha256": sha256_of(fp32_path),
            **meta,
        }
    )

    if do_quantize and target.quantize:
        int8_name = f"{target.key}_{target.imgsz}_uint8.onnx"
        int8_path = STATIC_MODELS_DIR / int8_name
        if quantize_uint8(fp32_path, int8_path):
            LOGGER.info("Wrote %s (%.2f MB)", int8_path.name, int8_path.stat().st_size / 1e6)
            target.produced.append(
                {
                    "file": int8_name,
                    "quantization": "uint8-dynamic",
                    "bytes": int8_path.stat().st_size,
                    "sha256": sha256_of(int8_path),
                    **describe_onnx(int8_path),
                }
            )

    return target


def build_manifest(targets: List[ExportTarget], imgsz: int, opset: int) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "generatedBy": "scripts/export_web_models.py",
        "exportImgsz": imgsz,
        "opset": opset,
        "layout": "NCHW",
        "pixelRange": "0-1",
        "letterbox": {"padValue": 114, "align": "center"},
        "decode": {
            "family": "yolo-v8-style",
            "description": (
                "Single output tensor [1, 4 + numClasses, numAnchors]. Rows 0-3 are "
                "cx, cy, w, h in input-pixel units; remaining rows are per-class "
                "sigmoid-free confidences already in 0-1."
            ),
        },
        "navigationClasses": NAVIGATION_CLASS_NAMES,
        "candidates": [
            {
                "key": t.key,
                "label": t.label,
                "imgsz": t.imgsz,
                "notes": t.notes,
                "artifacts": t.produced,
                "skipped": t.skipped_reason,
            }
            for t in targets
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imgsz", type=int, default=320, help="Square inference input size (default 320)")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset (default 13)")
    parser.add_argument("--no-quantize", action="store_true", help="Skip uint8 dynamic quantisation")
    parser.add_argument(
        "--models",
        nargs="*",
        default=["yolo11n", "yolo26n"],
        help="Candidate keys to export (default: yolo11n yolo26n)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    catalogue = {
        "yolo11n": ExportTarget(
            key="yolo11n",
            weights="yolo11n.pt",
            imgsz=args.imgsz,
            label="YOLO11n (nano)",
            notes="Candidate B. Vendored checkpoint in models/yolo11n.pt.",
        ),
        "yolo26n": ExportTarget(
            key="yolo26n",
            weights="yolo26n.pt",
            imgsz=args.imgsz,
            label="YOLO26n (nano)",
            notes="Candidate C. Exported only when the weights resolve in the installed ultralytics.",
        ),
    }

    unknown = [m for m in args.models if m not in catalogue]
    if unknown:
        parser.error(f"unknown candidate(s): {', '.join(unknown)}")

    targets = [process_target(catalogue[m], args.opset, not args.no_quantize) for m in args.models]

    manifest = build_manifest(targets, args.imgsz, args.opset)
    manifest_path = STATIC_MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", manifest_path)

    produced = sum(len(t.produced) for t in targets)
    for t in targets:
        if t.skipped_reason:
            LOGGER.info("  %-9s SKIPPED (%s)", t.key, t.skipped_reason.splitlines()[0][:90])
        else:
            for art in t.produced:
                LOGGER.info("  %-9s %-28s %7.2f MB", t.key, art["file"], art["bytes"] / 1e6)

    if produced == 0:
        LOGGER.error("No artifacts produced. The client will fall back to the CDN/MediaPipe candidates.")
        return 1

    LOGGER.info("Done: %d artifact(s) in %s", produced, STATIC_MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
