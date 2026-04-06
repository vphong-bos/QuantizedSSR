#!/usr/bin/env python3
"""
AIMET PyTorch PTQ script for the MMDet/MMCV detector flow shown in the user's
MMDet test script, rewritten to follow the structure and naming style of the
user's AIMET PyTorch code.

What is preserved from the detector code:
- config/plugin import flow
- build_dataset(cfg.data.test) + build_dataloader(...)
- checkpoint loading with build_model(...)
- DataContainer unpacking before inference
- detector invocation pattern: model(return_loss=False, rescale=True, **data)

What is borrowed from the AIMET PyTorch code style:
- parse_args() layout and option names
- explicit AIMET wrapper + forward callback functions
- optional CLE / BN fold / SeqMSE / QuantAnalyzer hooks
- QuantSim creation + compute_encodings + export flow

Notes:
- This is intentionally PyTorch/AIMET-first. For this detector stack, ONNX is
  usually harder because the forward path depends on rich metadata in the batch.
- AdaRound / bias correction are left as extension points unless your project
  already has helper functions equivalent to the segmentation example.
"""

import argparse
import copy
import os
import os.path as osp
import random
import sys
import time
import warnings
from typing import Any, Dict, Iterable, List, Optional

py_deps = os.environ.get("PY_DEPS_DIR")
if py_deps:
    if py_deps in sys.path:
        sys.path.remove(py_deps)
    sys.path.insert(0, py_deps)

sys.path.append("")

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model
from mmcv.utils import Registry, build_from_cfg

from mmdet.apis import set_random_seed
from mmdet.datasets import DATASETS, replace_ImageToTensor

from evaluation.eval_dataset import build_eval_loader
from quantization.quantize_function import AimetTraceWrapper, aimet_forward_fn, prepare_batch, create_quant_sim, calibration_forward_pass, move_to_device_keep_structure
from ssr.projects.mmdet3d_plugin.SSR.model import load_default_model

from aimet_common.defs import QuantScheme
from aimet_common.utils import CallbackFunc
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.seq_mse import SeqMseParams, apply_seq_mse
from aimet_torch import onnx as aimet_onnx
from aimet_torch import quantsim

from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.losses.iou_loss import GIoULoss
from mmcv.cnn.bricks.wrappers import Linear
from mmcv.cnn.bricks.drop import Dropout

from aimet_torch.v2.nn import QuantizationMixin

from evaluation.eval_dataset import extract_data

QuantizationMixin.ignore(FocalLoss)
QuantizationMixin.ignore(L1Loss)
QuantizationMixin.ignore(GIoULoss)
QuantizationMixin.ignore(Dropout)

from quantization.registered_ops import QuantizedLinear

warnings.filterwarnings("ignore")

OBJECTSAMPLERS = Registry("Object sampler")

import platform
from mmcv.utils import Registry

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# def analyzer_forward_pass(model, callback_args):
#     calib_loader, device, max_batches = callback_args
#     calibration_forward_pass(model, (calib_loader, device, max_batches))


# def analyzer_eval_callback(model, callback_args):
#     eval_loader, device, max_batches = callback_args
#     model.eval()

#     outputs = []
#     for batch_idx, batch in enumerate(eval_loader):
#         result = run_model_on_batch(model, batch, device)
#         if isinstance(result, list):
#             outputs.extend(result)
#         else:
#             outputs.append(result)
#         if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
#             break

#     # AIMET QuantAnalyzer expects a scalar score. We keep this generic by
#     # returning negative latency-free proxy if dataset.evaluate is unavailable.
#     dataset = eval_loader.dataset
#     try:
#         metrics = dataset.evaluate(outputs, metric=["bbox"])
#         if isinstance(metrics, dict) and metrics:
#             first_val = next(iter(metrics.values()))
#             return float(first_val)
#     except Exception as exc:
#         print(f"[WARN] analyzer_eval_callback dataset.evaluate failed: {exc}")

#     return float(len(outputs))

# -----------------------------------------------------------------------------
# AIMET stages
# -----------------------------------------------------------------------------
def maybe_fuse_conv_bn(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        print("Conv-BN fusion disabled")
        return model

    print("Applying mmcv fuse_conv_bn...")
    return fuse_conv_bn(model)


def maybe_run_bn_fold(wrapped_model: AimetTraceWrapper, dummy_input: torch.Tensor, enabled: bool) -> None:
    if not enabled:
        print("BN folding disabled")
        return

    print("Applying AIMET batch norm folding...")
    try:
        fold_all_batch_norms(
            model=wrapped_model,
            input_shapes=tuple(dummy_input.shape),
            dummy_input=dummy_input,
        )
    except TypeError:
        fold_all_batch_norms(
            model=wrapped_model,
            input_shapes=tuple(dummy_input.shape),
        )

def maybe_run_cle(wrapped_model: AimetTraceWrapper, dummy_input: torch.Tensor, enabled: bool) -> None:
    if not enabled:
        print("CLE disabled")
        return

    print("Applying Cross-Layer Equalization (CLE)...")
    cle_start = time.time()
    try:
        equalize_model(
            wrapped_model,
            input_shapes=tuple(dummy_input.shape),
            dummy_input=dummy_input,
        )
    except TypeError:
        equalize_model(wrapped_model, dummy_input=dummy_input)
    cle_time = time.time() - cle_start
    print(f"CLE finished in {cle_time:.2f} s")


def maybe_run_seq_mse(
    wrapped_model: AimetTraceWrapper,
    sim: QuantizationSimModel,
    calib_loader,
    enabled: bool,
    num_batches: int,
) -> None:
    if not enabled:
        print("Sequential MSE disabled")
        return

    print("Applying Sequential MSE...")
    params = SeqMseParams(
        num_batches=min(num_batches, len(calib_loader)),
        forward_fn=aimet_forward_fn,
    )

    try:
        apply_seq_mse(
            model=wrapped_model,
            sim=sim,
            data_loader=calib_loader,
            params=params,
            modules_to_exclude=None,
        )
    except TypeError:
        # Older AIMET variants may not accept sim/modules_to_exclude in the same way.
        apply_seq_mse(
            model=wrapped_model,
            dummy_input=None,
            data_loader=calib_loader,
            params=params,
            forward_fn=aimet_forward_fn,
        )

    print("Sequential MSE finished.")


# def maybe_run_quant_analyzer(
#     wrapped_model: AimetTraceWrapper,
#     dummy_input: torch.Tensor,
#     calib_loader,
#     enabled: bool,
#     quant_analyzer_dir: str,
#     analyzer_num_batches: Optional[int],
#     device: str,
#     eval_loader=None,
#     eval_max_batches: int = -1,
# ) -> None:
#     if not enabled:
#         return

#     print("Running AIMET QuantAnalyzer...")
#     os.makedirs(quant_analyzer_dir, exist_ok=True)

#     forward_pass_callback = CallbackFunc(
#         analyzer_forward_pass,
#         func_callback_args=(calib_loader, torch.device(device), analyzer_num_batches),
#     )

    # eval_callback = None
    # if eval_loader is not None:
    #     eval_callback = CallbackFunc(
    #         analyzer_eval_callback,
    #         func_callback_args=(eval_loader, torch.device(device), eval_max_batches),
    #     )

    # analyzer = QuantAnalyzer(
    #     model=wrapped_model,
    #     dummy_input=dummy_input,
    #     forward_pass_callback=forward_pass_callback,
    #     eval_callback=eval_callback,
    #     modules_to_ignore=None,
    # )

    # analyzer.analyze(
    #     quant_scheme="tf_enhanced",
    #     default_param_bw=8,
    #     default_output_bw=8,
    #     config_path=None,
    #     results_dir=quant_analyzer_dir,
    # )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default=None, help="SSR model's config")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work_dir", type=str, default="quantized_export")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)

    parser.add_argument("--quant_scheme", type=str, default="tf_enhanced", help="AIMET quantization scheme")
    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")
    parser.add_argument("--config_path", type=str, default=None, help="AIMET quantsim config file")

    parser.add_argument("--calib_batches", type=int, default=1, help="number of calibration batches")
    parser.add_argument("--eval_batches", type=int, default=-1, help="max eval batches, -1 means full set")
    parser.add_argument("--eval_metric", type=str, nargs="+", default=["bbox"], help="dataset.evaluate metrics")

    parser.add_argument("--run_fp32_eval", action="store_true", help="evaluate FP32 model")
    parser.add_argument("--run_int8_eval", action="store_true", help="evaluate quant sim model")

    parser.add_argument("--enable_cle", dest="enable_cle", action="store_true", help="enable CLE")
    parser.add_argument("--disable_cle", dest="enable_cle", action="store_false", help="disable CLE")
    parser.set_defaults(enable_cle=False)

    parser.add_argument("--enable_bn_fold", action="store_true", help="apply AIMET BN fold")
    parser.add_argument("--enable_seq_mse", action="store_true", help="apply AIMET SeqMSE")
    parser.add_argument("--seq_mse_num_batches", type=int, default=8, help="batches for SeqMSE")

    parser.add_argument("--fuse_conv_bn", action="store_true", help="apply mmcv fuse_conv_bn before AIMET")

    parser.add_argument("--run_quant_analyzer", action="store_true", help="run AIMET QuantAnalyzer")
    parser.add_argument("--quant_analyzer_dir", type=str, default="quant_analyzer_results")
    parser.add_argument("--analyzer_num_batches", type=int, default=None)

    parser.add_argument("--save_quant_checkpoint", type=str, default=None, help="optional torch save path for sim")
    parser.add_argument("--export_prefix", type=str, default="vad_detector_int8")
    parser.add_argument("--no_export", action="store_true", help="skip AIMET export")
    parser.add_argument("--export_onnx", action="store_true", help="also export AIMET QDQ ONNX")

    parser.add_argument(
        "--calib_max_samples",
        type=int,
        default=-1,
        help="maximum number of calibration samples, -1 means use calib_batches only"
    )

    return parser.parse_args(argv)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(args):
    os.makedirs(args.work_dir, exist_ok=True)

    if args.save_quant_checkpoint is not None:
        save_dir = os.path.dirname(args.save_quant_checkpoint)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    print("Building dataset / dataloader...")
    cfg, dataset, data_loader = build_eval_loader(args.config)

    print("Loading FP32 model...")
    model, _ = load_default_model(cfg, args.checkpoint, dataset, args.fuse_conv_bn, args.device)
    model = model.to(args.device).eval()

    import pickle

    def find_unpickleable(obj, prefix="root", seen=None):
        if seen is None:
            seen = set()

        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(obj, dict):
            for k, v in obj.items():
                find_unpickleable(v, f"{prefix}.{k}", seen)
            return

        if isinstance(obj, list):
            for i, v in enumerate(obj):
                find_unpickleable(v, f"{prefix}[{i}]", seen)
            return

        if isinstance(obj, tuple):
            for i, v in enumerate(obj):
                find_unpickleable(v, f"{prefix}({i})", seen)
            return

        try:
            pickle.dumps(obj)
        except Exception as e:
            print(prefix, type(obj), e)

    first_batch = next(iter(data_loader))
    first_batch = extract_data(first_batch)   # keep this function unchanged
    prepared_batch = move_to_device_keep_structure(first_batch, torch.device(args.device))

    wrapped_model = AimetTraceWrapper(model=model).to(args.device).eval()
    wrapped_model.set_batch(prepared_batch)

    real_img = prepared_batch["img"][0]   # because extracted format is [tensor]

    if not torch.is_tensor(real_img):
        raise TypeError(f"Expected tensor, got {type(real_img)}")

    # Ensure batch dimension exists
    if real_img.ndim == 4:   # [6, 3, H, W]
        real_img = real_img.unsqueeze(0)  # -> [1, 6, 3, H, W]
    elif real_img.ndim != 5:
        raise ValueError(f"Unexpected real_img shape: {real_img.shape}")

    print("trace real_img shape:", real_img.shape)

    dummy_input = torch.zeros_like(real_img)

    maybe_run_cle(wrapped_model, dummy_input, args.enable_cle)
    maybe_run_bn_fold(wrapped_model, dummy_input, args.enable_bn_fold)

    # maybe_run_quant_analyzer(
    #     wrapped_model=wrapped_model,
    #     dummy_input=dummy_input,
    #     calib_loader=data_loader,
    #     enabled=args.run_quant_analyzer,
    #     quant_analyzer_dir=osp.join(args.work_dir, args.quant_analyzer_dir),
    #     analyzer_num_batches=args.analyzer_num_batches,
    #     device=args.device,
    #     eval_loader=data_loader if args.run_fp32_eval else None,
    #     eval_max_batches=args.eval_batches,
    # )

    # skip_layer_names = [
    #     # "pts_bbox_head.positional_encoding.*",
    #     # "pts_bbox_head.transformer.level_embeds",
    #     # "pts_bbox_head.transformer.cams_embeds",
    #     # "pts_bbox_head.bev_embedding.weight",
    #     # "pts_bbox_head.query_embedding.weight",
    #     # "pts_bbox_head.map_instance_embedding.weight",
    #     # "pts_bbox_head.map_pts_embedding.weight",
    #     # "pts_bbox_head.ego_query.weight",
    #     # "pts_bbox_head.navi_embedding.weight",
    #     # "pts_bbox_head.way_point.weight",
    # ]

    def should_skip(name: str) -> bool:
        return (
            # whole transformer stack
            "pts_bbox_head.transformer" in name
            or "transformer.encoder.layers" in name
            or "attentions" in name
            or "deformable_attention" in name
            or "ffns" in name

            # transformer-adjacent / geometric / positional
            or "positional_encoding" in name
            or "reference_points" in name
            or "map_reference_points" in name
            or "level_embeds" in name
            or "cams_embeds" in name

            # embeddings / learned queries
            or "bev_embedding" in name
            or "query_embedding" in name
            or "map_instance_embedding" in name
            or "map_pts_embedding" in name
            or "ego_query" in name
            or "navi_embedding" in name
            or "way_point" in name
            or "embedding" in name

            # norms
            or ".norm" in name
            or ".norms." in name
            or "layer_norm" in name

            # fragile side branches
            or "can_bus_mlp" in name
            or "navi_se" in name
            or "tokenlearner" in name
            or "tokenfuser" in name
            or "latent_decoder" in name
            or "way_decoder" in name
            or "latent_world_model" in name
            or "action_mln" in name
            or "pos_mln" in name

            # optional: keep all planning / traj / map heads float for maximum safety
            or "traj_branches" in name
            or "traj_cls_branches" in name
            or "map_cls_branches" in name
            or "map_reg_branches" in name
            or "ego_fut_decoder" in name
        )

    def get_skip_layer_names(model):
        skip_layer_names = []

        for name, module in model.named_modules():
            if should_skip(name):
                skip_layer_names.append(name)

        skip_layer_names = list(dict.fromkeys(skip_layer_names))
        return skip_layer_names

    # skip_layer_names = get_skip_layer_names(wrapped_model)

    skip_layer_names = []

    # skip_layer_names.extend([
    #     "model.pts_bbox_head.positional_encoding",
    #     "model.pts_bbox_head.bev_embedding",
    #     "model.pts_bbox_head.query_embedding",
    #     "model.pts_bbox_head.map_instance_embedding",
    #     "model.pts_bbox_head.map_pts_embedding",
    #     "model.pts_bbox_head.ego_query",
    #     "model.pts_bbox_head.navi_embedding",

    #     "model.pts_bbox_head.transformer.reference_points",
    #     "model.pts_bbox_head.transformer.map_reference_points",

    #     "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.attention_weights",
    #     "model.pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.attention_weights",
    #     "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.attention_weights",
    #     "model.pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.attention_weights",
    #     "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.attention_weights",
    #     "model.pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.sampling_offsets",
    #     "model.pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.attention_weights",
    # ])

    print("Creating AIMET QuantizationSimModel...")
    sim = create_quant_sim(
        model=wrapped_model,
        device=args.device,
        dummy_input=dummy_input,
        quant_scheme=args.quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_path=args.config_path,
        skip_layer_names=skip_layer_names
    )

    if args.enable_seq_mse:
        maybe_run_seq_mse(
            wrapped_model=wrapped_model,
            sim=sim,
            calib_loader=data_loader,
            enabled=True,
            num_batches=args.seq_mse_num_batches,
        )
    else:
        print("Sequential MSE disabled")

    print("Computing encodings with calibration data...")
    calib_start = time.time()
    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(
            data_loader,
            torch.device(args.device),
            args.calib_batches,
            args.calib_max_samples,
        ),
    )
    calib_time = time.time() - calib_start
    print(f"Calibration finished in {calib_time:.2f} s")

    # import cv2
    # import pickle
    # from collections.abc import Mapping, Sequence

    # def find_objects_by_type(obj, target_type, max_depth=8):
    #     visited = set()
    #     hits = []

    #     def walk(x, path, depth):
    #         if depth > max_depth:
    #             return

    #         obj_id = id(x)
    #         if obj_id in visited:
    #             return
    #         visited.add(obj_id)

    #         try:
    #             if isinstance(x, target_type):
    #                 hits.append(path)
    #                 return
    #         except Exception:
    #             pass

    #         # dict-like
    #         if isinstance(x, Mapping):
    #             for k, v in x.items():
    #                 walk(v, f"{path}[{k!r}]", depth + 1)
    #             return

    #         # list/tuple-like, but avoid strings/bytes
    #         if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
    #             for i, v in enumerate(x):
    #                 walk(v, f"{path}[{i}]", depth + 1)
    #             return

    #         # normal python object attributes
    #         if hasattr(x, "__dict__"):
    #             for name, v in vars(x).items():
    #                 walk(v, f"{path}.{name}", depth + 1)

    #     walk(obj, "root", 0)
    #     return hits


    # def find_non_picklable_paths(obj, max_depth=6):
    #     visited = set()
    #     bad = []

    #     def walk(x, path, depth):
    #         if depth > max_depth:
    #             return

    #         obj_id = id(x)
    #         if obj_id in visited:
    #             return
    #         visited.add(obj_id)

    #         # Try pickling this object itself
    #         try:
    #             pickle.dumps(x)
    #         except Exception as e:
    #             bad.append((path, type(x).__name__, repr(e)))

    #         if isinstance(x, Mapping):
    #             for k, v in x.items():
    #                 walk(v, f"{path}[{k!r}]", depth + 1)
    #             return

    #         if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
    #             for i, v in enumerate(x):
    #                 walk(v, f"{path}[{i}]", depth + 1)
    #             return

    #         if hasattr(x, "__dict__"):
    #             for name, v in vars(x).items():
    #                 walk(v, f"{path}.{name}", depth + 1)

    #     walk(obj, "root", 0)
    #     return bad

    # vw_hits = find_objects_by_type(sim, cv2.VideoWriter, max_depth=10)
    # print("VideoWriter hits:")
    # for p in vw_hits:
    #     print("  ", p)

    # bad_hits = find_non_picklable_paths(sim, max_depth=5)
    # print("\nNon-picklable hits:")
    # for path, typ, err in bad_hits[:100]:
    #     print(f"{path} :: {typ} :: {err}")

    # def remove_video_writers(obj, visited=None):
    #     if visited is None:
    #         visited = set()

    #     obj_id = id(obj)
    #     if obj_id in visited:
    #         return
    #     visited.add(obj_id)

    #     # Remove known attributes
    #     for attr in ["_video_writers", "_combined_video_writers", "_integrated_video_writers"]:
    #         if hasattr(obj, attr):
    #             setattr(obj, attr, None)

    #     # Traverse children
    #     if hasattr(obj, "__dict__"):
    #         for v in vars(obj).values():
    #             remove_video_writers(v, visited)

    #     elif isinstance(obj, (list, tuple)):
    #         for v in obj:
    #             remove_video_writers(v, visited)

    #     elif isinstance(obj, dict):
    #         for v in obj.values():
    #             remove_video_writers(v, visited)


    # # Apply to full sim (important!)
    # remove_video_writers(sim)

    save_dir = args.save_quant_checkpoint
    os.makedirs(save_dir, exist_ok=True)

    # unwrap if needed
    base_model = sim.model
    if hasattr(base_model, "model"):
        base_model = base_model.model

    # save weights
    torch.save(base_model.state_dict(), os.path.join(save_dir, "model_state_dict.pth"))

    # # save AIMET encodings for deployment / reload
    # sim.export(
    #     path=save_dir,
    #     filename_prefix="quantized_ssr",
    #     dummy_input=dummy_input,
    # )

    print(f"Saved model weights and encodings to {save_dir}")

    # if args.save_quant_checkpoint is not None:
    #     quantsim.save_checkpoint(sim, args.save_quant_checkpoint)
    #     print(f"Saved AIMET sim checkpoint to: {args.save_quant_checkpoint}")

    if not args.no_export:
        export_dir = osp.join(args.work_dir, args.export_prefix)
        os.makedirs(export_dir, exist_ok=True)

        print(f"Exporting AIMET artifacts to: {export_dir}")
        sim.export(
            path=export_dir,
            filename_prefix=args.export_prefix,
            dummy_input=dummy_input,
        )

    if args.export_onnx:
        export_dir = osp.join(args.work_dir, args.export_prefix)
        onnx_path = osp.join(export_dir, f"{args.export_prefix}.onnx")
        print("Exporting quantized model to ONNX QDQ...")
        aimet_onnx.export(
            sim.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=20,
            export_int32_bias=True,
            prequantize_constants=True
        )
        print(f"Exported QDQ ONNX to: {onnx_path}")

    rank, _ = get_dist_info()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
