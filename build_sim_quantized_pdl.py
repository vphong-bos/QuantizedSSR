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

from ssr.projects.mmdet3d_plugin.datasets.builder import build_dataloader
from ssr.projects.mmdet3d_plugin.SSR.utils.builder import build_model

from aimet_common.defs import QuantScheme
from aimet_common.utils import CallbackFunc
from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quant_analyzer import QuantAnalyzer
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.seq_mse import SeqMseParams, apply_seq_mse
from aimet_torch import onnx as aimet_onnx

from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.losses.iou_loss import GIoULoss
from mmcv.cnn.bricks.wrappers import Linear
from mmcv.cnn.bricks.drop import Dropout

from aimet_torch.v2.nn import QuantizationMixin

QuantizationMixin.ignore(FocalLoss)
QuantizationMixin.ignore(L1Loss)
QuantizationMixin.ignore(GIoULoss)
QuantizationMixin.ignore(Dropout)

@QuantizationMixin.implements(Linear)
class QuantizedLinear(QuantizationMixin, Linear):

    def __quant_init__(self):
        super().__quant_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

warnings.filterwarnings("ignore")

OBJECTSAMPLERS = Registry("Object sampler")


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args)


# -----------------------------------------------------------------------------
# Data handling from the detector script
# -----------------------------------------------------------------------------
def extract_data_from_container(data: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(data)
    data["img_metas"] = data["img_metas"][0].data
    data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data
    data["gt_labels_3d"] = data["gt_labels_3d"][0].data
    data["img"] = data["img"][0].data
    data["ego_his_trajs"] = data["ego_his_trajs"][0].data
    data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data
    data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data
    data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data
    data["gt_attr_labels"] = data["gt_attr_labels"][0].data
    data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
    data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
    return data


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch = extract_data_from_container(batch)
    batch = move_to_device(batch, device)
    return batch


# -----------------------------------------------------------------------------
# AIMET wrappers / callbacks in the user's style
# -----------------------------------------------------------------------------
# class AimetTraceWrapper(torch.nn.Module):
#     """
#     Wrap the detector so AIMET sees a conventional forward(img) API.

#     The rest of the batch metadata is cached from the latest dataloader sample.
#     This mirrors your segmentation-side AimetTraceWrapper idea, but adapted to
#     detector inference where img is only one field of a larger batch dict.
#     """

#     def __init__(self, model: torch.nn.Module, initial_batch: Dict[str, Any]):
#         super().__init__()
#         self.model = model
#         self.runtime_batch = initial_batch

#     def set_batch(self, batch: Dict[str, Any]) -> None:
#         self.runtime_batch = batch

#     def forward(self, images):
#         batch = dict(self.runtime_batch)
#         batch["img"] = images
#         out = self.model(return_loss=False, rescale=True, **batch)
#         return self._make_traceable_output(out)

#     @staticmethod
#     def _make_traceable_output(out: Any):
#         if torch.is_tensor(out):
#             return out

#         if isinstance(out, dict):
#             tensor_dict = {k: v for k, v in out.items() if torch.is_tensor(v)}
#             if len(tensor_dict) == 1:
#                 return next(iter(tensor_dict.values()))
#             if tensor_dict:
#                 return tensor_dict

#         if isinstance(out, (list, tuple)):
#             gathered = []
#             for item in out:
#                 if torch.is_tensor(item):
#                     gathered.append(item)
#                 elif isinstance(item, dict):
#                     for value in item.values():
#                         if torch.is_tensor(value):
#                             gathered.append(value)
#             if len(gathered) == 1:
#                 return gathered[0]
#             if gathered:
#                 return tuple(gathered)

#         raise RuntimeError(
#             "Model output does not expose tensors suitable for AIMET tracing. "
#             "You may need to wrap a deeper internal submodule for this model."
#         )


# def aimet_forward_fn(model, inputs):
#     if isinstance(inputs, dict):
#         images = inputs["img"]
#     elif isinstance(inputs, (list, tuple)):
#         images = inputs[0]
#     else:
#         images = inputs

#     images = images.to(next(model.parameters()).device)
#     return model(images)

class AimetTraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.runtime_batch = None

    def set_batch(self, batch):
        self.runtime_batch = batch

        print(batch)

    def forward(self, _dummy):
        batch = self.runtime_batch
        assert batch is not None, "Batch not set"

        img = batch["img"]
        img_metas = batch["img_metas"]

        # 🔥 Use feature extractor instead of full inference
        feats = self.model.extract_feat(img=img, img_metas=img_metas)

        # AIMET needs a tensor → pick one
        if isinstance(feats, (list, tuple)):
            for f in feats:
                if torch.is_tensor(f):
                    return f
        elif torch.is_tensor(feats):
            return feats

        raise RuntimeError("extract_feat did not return tensor")
    
    @staticmethod
    def _make_traceable_output(out):
        if torch.is_tensor(out):
            return out

        if isinstance(out, dict):
            tensor_dict = {k: v for k, v in out.items() if torch.is_tensor(v)}
            if len(tensor_dict) == 1:
                return next(iter(tensor_dict.values()))
            if tensor_dict:
                return tensor_dict

        if isinstance(out, (list, tuple)):
            gathered = []
            for item in out:
                if torch.is_tensor(item):
                    gathered.append(item)
                elif isinstance(item, dict):
                    for value in item.values():
                        if torch.is_tensor(value):
                            gathered.append(value)
            if len(gathered) == 1:
                return gathered[0]
            if gathered:
                return tuple(gathered)

        raise RuntimeError("Model output does not expose tensors suitable for AIMET tracing.")

def aimet_forward_fn(model, inputs):
    return model(torch.zeros(1, device=next(model.parameters()).device))


@torch.no_grad()
def run_model_on_batch(model: AimetTraceWrapper, batch: Dict[str, Any], device: torch.device):
    batch = prepare_batch(batch, device)
    model.set_batch(batch)
    return model(batch["img"])


@torch.no_grad()
def calibration_forward_pass(model, callback_args):
    calib_loader, device, max_batches = callback_args

    model.eval()
    for batch_idx, batch in enumerate(calib_loader):
        _ = run_model_on_batch(model, batch, device)
        if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
            break


def analyzer_forward_pass(model, callback_args):
    calib_loader, device, max_batches = callback_args
    calibration_forward_pass(model, (calib_loader, device, max_batches))


def analyzer_eval_callback(model, callback_args):
    eval_loader, device, max_batches = callback_args
    model.eval()

    outputs = []
    for batch_idx, batch in enumerate(eval_loader):
        result = run_model_on_batch(model, batch, device)
        if isinstance(result, list):
            outputs.extend(result)
        else:
            outputs.append(result)
        if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
            break

    # AIMET QuantAnalyzer expects a scalar score. We keep this generic by
    # returning negative latency-free proxy if dataset.evaluate is unavailable.
    dataset = eval_loader.dataset
    try:
        metrics = dataset.evaluate(outputs, metric=["bbox"])
        if isinstance(metrics, dict) and metrics:
            first_val = next(iter(metrics.values()))
            return float(first_val)
    except Exception as exc:
        print(f"[WARN] analyzer_eval_callback dataset.evaluate failed: {exc}")

    return float(len(outputs))


# -----------------------------------------------------------------------------
# Build helpers from the detector script
# -----------------------------------------------------------------------------
def import_plugin_modules(cfg: Config, config_path: str) -> None:
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg["custom_imports"])

    if hasattr(cfg, "plugin") and cfg.plugin:
        import importlib
        if hasattr(cfg, "plugin_dir"):
            module_dir = os.path.dirname(cfg.plugin_dir).split("/")
        else:
            module_dir = os.path.dirname(config_path).split("/")

        module_path = module_dir[0]
        for m in module_dir[1:]:
            module_path = module_path + "." + m
        print("[INFO] Importing plugin module:", module_path)
        importlib.import_module(module_path)


def build_dataset_and_loader(cfg: Config):
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(ds_cfg.pop("samples_per_gpu", 1) for ds_cfg in cfg.data.test)
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return dataset, data_loader


def build_fp32_model(cfg: Config, checkpoint_path: str):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    if "CLASSES" in checkpoint.get("meta", {}):
        model.CLASSES = checkpoint["meta"]["CLASSES"]
    if "PALETTE" in checkpoint.get("meta", {}):
        model.PALETTE = checkpoint["meta"]["PALETTE"]
    return model


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def run_eval(model: AimetTraceWrapper, data_loader, device: torch.device, max_batches: int = -1):
    model.eval()
    outputs = []
    total = len(data_loader.dataset) if max_batches < 0 else min(len(data_loader.dataset), max_batches)
    prog_bar = mmcv.ProgressBar(total)

    for batch_idx, batch in enumerate(data_loader):
        result = run_model_on_batch(model, batch, device)
        if isinstance(result, list):
            outputs.extend(result)
            batch_size = len(result)
        else:
            outputs.append(result)
            batch_size = 1

        for _ in range(batch_size):
            prog_bar.update()

        if max_batches > 0 and batch_idx + 1 >= max_batches:
            break

    return outputs


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


def maybe_run_quant_analyzer(
    wrapped_model: AimetTraceWrapper,
    dummy_input: torch.Tensor,
    calib_loader,
    enabled: bool,
    quant_analyzer_dir: str,
    analyzer_num_batches: Optional[int],
    device: str,
    eval_loader=None,
    eval_max_batches: int = -1,
) -> None:
    if not enabled:
        return

    print("Running AIMET QuantAnalyzer...")
    os.makedirs(quant_analyzer_dir, exist_ok=True)

    forward_pass_callback = CallbackFunc(
        analyzer_forward_pass,
        func_callback_args=(calib_loader, torch.device(device), analyzer_num_batches),
    )

    eval_callback = None
    if eval_loader is not None:
        eval_callback = CallbackFunc(
            analyzer_eval_callback,
            func_callback_args=(eval_loader, torch.device(device), eval_max_batches),
        )

    analyzer = QuantAnalyzer(
        model=wrapped_model,
        dummy_input=dummy_input,
        forward_pass_callback=forward_pass_callback,
        eval_callback=eval_callback,
        modules_to_ignore=None,
    )

    analyzer.analyze(
        quant_scheme="tf_enhanced",
        default_param_bw=8,
        default_output_bw=8,
        config_file=None,
        results_dir=quant_analyzer_dir,
    )


def create_quant_sim(
    model: AimetTraceWrapper,
    device: str,
    dummy_input: torch.Tensor,
    quant_scheme: str,
    default_output_bw: int,
    default_param_bw: int,
    config_file: Optional[str],
):
    scheme_map = {
        "tf": QuantScheme.post_training_tf,
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    }
    selected_scheme = scheme_map.get(quant_scheme, QuantScheme.post_training_tf_enhanced)

    sim = QuantizationSimModel(
        model=model.to(device).eval(),
        dummy_input=dummy_input,
        quant_scheme=selected_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file=config_file,
        in_place=False,
    )
    return sim


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="test config file path")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work_dir", type=str, default="quantized_export")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)

    parser.add_argument("--quant_scheme", type=str, default="tf_enhanced", help="AIMET quantization scheme")
    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")
    parser.add_argument("--config_file", type=str, default=None, help="AIMET quantsim config file")

    parser.add_argument("--calib_batches", type=int, default=32, help="number of calibration batches")
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

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    import_plugin_modules(cfg, args.config)

    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    print("Building dataset / dataloader...")
    dataset, data_loader = build_dataset_and_loader(copy.deepcopy(cfg))

    print("Loading FP32 model...")
    model = build_fp32_model(cfg, args.checkpoint)
    model = maybe_fuse_conv_bn(model, args.fuse_conv_bn)
    model = model.to(args.device).eval()

    first_batch = next(iter(data_loader))
    prepared_batch = prepare_batch(first_batch, torch.device(args.device))
    dummy_input = prepared_batch["img"]

    # print("Wrapping model for AIMET tracing...")
    # wrapped_model = AimetTraceWrapper(model=model, initial_batch=prepared_batch).to(args.device).eval()

    wrapped_model = AimetTraceWrapper(model=model).to(args.device).eval()
    wrapped_model.set_batch(prepared_batch)

    # dummy input is irrelevant now
    dummy_input = torch.zeros(1, device=args.device)

    maybe_run_cle(wrapped_model, dummy_input, args.enable_cle)
    maybe_run_bn_fold(wrapped_model, dummy_input, args.enable_bn_fold)

    if args.run_fp32_eval:
        print("Running FP32 evaluation...")
        fp32_outputs = run_eval(wrapped_model, data_loader, torch.device(args.device), args.eval_batches)
        fp32_metrics = dataset.evaluate(fp32_outputs, metric=args.eval_metric)
        print("[FP32]", fp32_metrics)

    maybe_run_quant_analyzer(
        wrapped_model=wrapped_model,
        dummy_input=dummy_input,
        calib_loader=data_loader,
        enabled=args.run_quant_analyzer,
        quant_analyzer_dir=osp.join(args.work_dir, args.quant_analyzer_dir),
        analyzer_num_batches=args.analyzer_num_batches,
        device=args.device,
        eval_loader=data_loader if args.run_fp32_eval else None,
        eval_max_batches=args.eval_batches,
    )

    print("Creating AIMET QuantizationSimModel...")
    sim = create_quant_sim(
        model=wrapped_model,
        device=args.device,
        dummy_input=dummy_input,
        quant_scheme=args.quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file=args.config_file,
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
        forward_pass_callback_args=(data_loader, torch.device(args.device), args.calib_batches),
    )
    calib_time = time.time() - calib_start
    print(f"Calibration finished in {calib_time:.2f} s")

    if args.run_int8_eval:
        print("Running INT8-sim evaluation...")
        int8_outputs = run_eval(sim.model, data_loader, torch.device(args.device), args.eval_batches)
        int8_metrics = dataset.evaluate(int8_outputs, metric=args.eval_metric)
        print("[INT8]", int8_metrics)

    if args.save_quant_checkpoint is not None:
        torch.save(sim, args.save_quant_checkpoint)
        print(f"Saved AIMET sim checkpoint to: {args.save_quant_checkpoint}")

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
                prequantize_constants=True,
                dynamo=False,
            )
            print(f"Exported QDQ ONNX to: {onnx_path}")
    else:
        print("Export disabled")

    rank, _ = get_dist_info()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
