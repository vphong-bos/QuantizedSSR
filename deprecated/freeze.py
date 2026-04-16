# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Phong Vu
# ---------------------------------------------
import os
import sys
import json
import shutil
import hashlib
from datetime import datetime

py_deps = os.environ.get("PY_DEPS_DIR")
if py_deps:
    if py_deps in sys.path:
        sys.path.remove(py_deps)
    sys.path.insert(0, py_deps)

sys.path.append("")

print("PY_DEPS_DIR =", py_deps)
print("sys.path[:6] =", sys.path[:6])

import argparse
import numpy as np
import torch
import mmcv
print("mmcv loaded from:", mmcv.__file__)
import warnings

from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.losses.iou_loss import GIoULoss
from mmcv.cnn.bricks.drop import Dropout

from aimet_torch.v2.nn import QuantizationMixin
from quantization.registered_ops import QuantizedLinear
from quantization.quantize_function import load_quantized_model

import onnx
from onnx import numpy_helper

warnings.filterwarnings("ignore")

QuantizationMixin.ignore(FocalLoss)
QuantizationMixin.ignore(L1Loss)
QuantizationMixin.ignore(GIoULoss)
QuantizationMixin.ignore(Dropout)


QUANT_OP_TYPES_EXACT = {
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearAveragePool",
    "QLinearGlobalAveragePool",
    "QLinearLeakyRelu",
    "QLinearSigmoid",
    "QLinearSoftmax",
    "ConvInteger",
    "MatMulInteger",
}

QUANT_OP_KEYWORDS = (
    "QuantizeLinear",
    "DequantizeLinear",
    "QLinear",
    "Integer",
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def sha256_file(path):
    if path is None or not os.path.isfile(path):
        return None

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def copy_into_package(src_path, dst_dir):
    if not src_path:
        return None

    ensure_dir(dst_dir)
    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    shutil.copy2(src_path, dst_path)

    if src_path.endswith(".onnx"):
        src_data = src_path + "_data"
        dst_data = dst_path + "_data"
        if os.path.exists(src_data):
            shutil.copy2(src_data, dst_data)

    return dst_path


def summarize_array(arr):
    arr = np.asarray(arr)
    info = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
    }
    if arr.size > 0 and arr.dtype.kind in ("f", "i", "u"):
        info.update({
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        })
    return info


def is_quant_op(op_type):
    if op_type in QUANT_OP_TYPES_EXACT:
        return True
    return any(keyword in op_type for keyword in QUANT_OP_KEYWORDS)


def collect_onnx_op_inventory(model_path):
    model = onnx.load(model_path, load_external_data=False)

    op_counts = {}
    quant_op_counts = {}
    non_quant_op_counts = {}
    node_rows = []

    total_nodes = len(model.graph.node)
    quant_node_count = 0

    for idx, node in enumerate(model.graph.node):
        op_type = node.op_type
        op_counts[op_type] = op_counts.get(op_type, 0) + 1

        quant_flag = is_quant_op(op_type)
        if quant_flag:
            quant_op_counts[op_type] = quant_op_counts.get(op_type, 0) + 1
            quant_node_count += 1
        else:
            non_quant_op_counts[op_type] = non_quant_op_counts.get(op_type, 0) + 1

        node_rows.append({
            "index": idx,
            "name": node.name,
            "op_type": op_type,
            "is_quant_op": quant_flag,
            "inputs": list(node.input),
            "outputs": list(node.output),
        })

    return {
        "node_count": total_nodes,
        "quant_node_count": quant_node_count,
        "non_quant_node_count": total_nodes - quant_node_count,
        "quant_node_ratio": (float(quant_node_count) / float(total_nodes)) if total_nodes > 0 else 0.0,
        "op_counts": dict(sorted(op_counts.items(), key=lambda x: x[0])),
        "quant_op_counts": dict(sorted(quant_op_counts.items(), key=lambda x: x[0])),
        "non_quant_op_counts": dict(sorted(non_quant_op_counts.items(), key=lambda x: x[0])),
        "all_ops": sorted(op_counts.keys()),
        "quant_ops": sorted(quant_op_counts.keys()),
        "non_quant_ops": sorted(non_quant_op_counts.keys()),
        "nodes": node_rows,
    }


def collect_onnx_qparams_and_tensor_metadata(model_path):
    model = onnx.load(model_path, load_external_data=False)
    qparams = []
    tensor_meta = []

    init_map = {init.name: init for init in model.graph.initializer}

    def initializer_to_full_json(init_obj):
        info = {
            "name": init_obj.name,
            "dims": list(init_obj.dims),
            "data_type": int(init_obj.data_type),
            "uses_external_data": bool(init_obj.external_data),
        }

        if not init_obj.external_data:
            try:
                arr = numpy_helper.to_array(init_obj)
                info["dtype"] = str(arr.dtype)
                info["value"] = arr.tolist()
                info["summary"] = summarize_array(arr)
            except Exception as e:
                info["value_error"] = str(e)

        return info

    for init in model.graph.initializer:
        meta_row = {
            "name": init.name,
            "dims": list(init.dims),
            "data_type": int(init.data_type),
            "uses_external_data": bool(init.external_data),
        }

        if not init.external_data:
            try:
                arr = numpy_helper.to_array(init)
                meta_row["dtype"] = str(arr.dtype)
                meta_row["summary"] = summarize_array(arr)
            except Exception as e:
                meta_row["summary_error"] = str(e)

        tensor_meta.append(meta_row)

    for node in model.graph.node:
        if is_quant_op(node.op_type):
            axis = None
            for attr in node.attribute:
                if attr.name == "axis":
                    axis = int(attr.i)

            entry = {
                "node_name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "axis": axis,
            }

            if len(node.input) >= 2 and node.input[1] in init_map:
                entry["scale"] = initializer_to_full_json(init_map[node.input[1]])

            if len(node.input) >= 3 and node.input[2] in init_map:
                entry["zero_point"] = initializer_to_full_json(init_map[node.input[2]])

            qparams.append(entry)

    return {
        "qparam_nodes": qparams,
        "initializer_tensor_metadata": tensor_meta,
    }


def compare_onnx_graphs(fp32_onnx_path, quant_onnx_path):
    if not fp32_onnx_path or not quant_onnx_path:
        return None

    fp32_inv = collect_onnx_op_inventory(fp32_onnx_path)
    quant_inv = collect_onnx_op_inventory(quant_onnx_path)

    fp32_ops = fp32_inv["op_counts"]
    quant_ops = quant_inv["op_counts"]

    all_ops = sorted(set(fp32_ops.keys()) | set(quant_ops.keys()))
    diff_rows = []

    for op in all_ops:
        diff_rows.append({
            "op_type": op,
            "fp32_count": fp32_ops.get(op, 0),
            "quant_count": quant_ops.get(op, 0),
            "delta": quant_ops.get(op, 0) - fp32_ops.get(op, 0),
            "is_quant_op": is_quant_op(op),
        })

    return {
        "fp32_node_count": fp32_inv["node_count"],
        "quant_node_count": quant_inv["node_count"],
        "fp32_ops": fp32_inv["all_ops"],
        "quant_ops": quant_inv["all_ops"],
        "quant_only_ops": sorted(set(quant_inv["all_ops"]) - set(fp32_inv["all_ops"])),
        "fp32_only_ops": sorted(set(fp32_inv["all_ops"]) - set(quant_inv["all_ops"])),
        "op_count_diff": diff_rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export quantized model op inventory")
    parser.add_argument("--config_path", help="config for sim model quantization")
    parser.add_argument("--quant_weights", required=True, help="quantized checkpoint / onnx file")
    parser.add_argument("--encoding_path", default=None, help="(Optional) Encoding file")
    parser.add_argument("--config", default=None, help="Config file for model creation")
    parser.add_argument("--enable_bn_fold", action="store_true", help="apply AIMET BN fold")

    parser.add_argument(
        "--graph_optimization_level",
        choices=["disable_all", "basic", "extended", "all"],
        default="disable_all",
        help="Quantization optimization level, only works with onnx format model",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Choose device",
    )
    parser.add_argument(
        "--provider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        default="CPUExecutionProvider",
        help="Choose onnx provider",
    )

    parser.add_argument("--package_root", type=str, default="handoff_packages")
    parser.add_argument("--artifact_name", type=str, default="quant_op_inventory")
    parser.add_argument("--artifact_version", type=str, default="v1")
    parser.add_argument("--fp32_onnx", type=str, default=None,
                        help="Optional FP32 ONNX path for graph diff")
    return parser.parse_args()


def export_quant_op_inventory(args):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    package_dir = os.path.join(
        args.package_root,
        "{}_{}_{}".format(args.artifact_name, args.artifact_version, timestamp),
    )

    models_dir = os.path.join(package_dir, "models")
    notes_dir = os.path.join(package_dir, "notes")
    ensure_dir(models_dir)
    ensure_dir(notes_dir)

    print("Loading quantized model from args.quant_weights ...")
    quant_obj = load_quantized_model(
        quant_weights=args.quant_weights,
        device=args.device,
        graph_optimization_level=args.graph_optimization_level,
        provider=args.provider,
        encoding_path=args.encoding_path,
        config=args.config,
        config_path=args.config_path,
        enable_bn_fold=args.enable_bn_fold,
    )
    print("Loaded quant backend:", quant_obj.get("backend"))

    packaged_quant_model = copy_into_package(args.quant_weights, models_dir)
    packaged_fp32_onnx = copy_into_package(args.fp32_onnx, models_dir) if args.fp32_onnx else None

    quant_op_inventory = {
        "warning": "Quant model is not ONNX. ONNX op inventory unavailable."
    }
    quant_qparams = {
        "warning": "Quant model is not ONNX. QParam extraction unavailable."
    }
    graph_diff = None

    quant_model_packaged_path = os.path.join(models_dir, os.path.basename(args.quant_weights))
    if quant_model_packaged_path.endswith(".onnx"):
        print("Extracting ONNX op inventory ...")
        quant_op_inventory = collect_onnx_op_inventory(quant_model_packaged_path)
        quant_qparams = collect_onnx_qparams_and_tensor_metadata(quant_model_packaged_path)

        if packaged_fp32_onnx:
            fp32_onnx_packaged_path = os.path.join(models_dir, os.path.basename(args.fp32_onnx))
            graph_diff = compare_onnx_graphs(fp32_onnx_packaged_path, quant_model_packaged_path)

    manifest = {
        "artifact": {
            "name": args.artifact_name,
            "version": args.artifact_version,
            "created_at_utc": timestamp,
            "package_dir": package_dir,
        },
        "verification": {
            "quant_load_ok": True,
            "quant_backend": quant_obj.get("backend"),
        },
        "models": {
            "quant_model": os.path.basename(packaged_quant_model) if packaged_quant_model else None,
            "fp32_onnx": os.path.basename(packaged_fp32_onnx) if packaged_fp32_onnx else None,
        },
        "checksums": {
            "quant_model_sha256": sha256_file(packaged_quant_model) if packaged_quant_model else None,
            "fp32_onnx_sha256": sha256_file(packaged_fp32_onnx) if packaged_fp32_onnx else None,
        },
    }

    save_json(manifest, os.path.join(package_dir, "manifest.json"))
    save_json(quant_op_inventory, os.path.join(notes_dir, "op_inventory.json"))
    save_json(quant_qparams, os.path.join(notes_dir, "quant_assets.json"))
    if graph_diff is not None:
        save_json(graph_diff, os.path.join(notes_dir, "graph_diff.json"))

    print("Done.")
    print("Package dir:", package_dir)
    print("Op inventory:", os.path.join(notes_dir, "op_inventory.json"))

    return package_dir


def main():
    args = parse_args()
    export_quant_op_inventory(args)


if __name__ == "__main__":
    main()