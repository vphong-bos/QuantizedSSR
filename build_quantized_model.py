import argparse
import os
import onnxsim
import os.path as osp
import random
import sys
import time
import warnings

py_deps = os.environ.get("PY_DEPS_DIR")
if py_deps:
    if py_deps in sys.path:
        sys.path.remove(py_deps)
    sys.path.insert(0, py_deps)

sys.path.append("")

import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import get_dist_info
from mmcv.utils import Registry

from mmdet.apis import set_random_seed

from aimet_torch import onnx as aimet_onnx

import traceback
from contextlib import contextmanager


from mmdet.models.losses.focal_loss import FocalLoss
from mmdet.models.losses.smooth_l1_loss import L1Loss
from mmdet.models.losses.iou_loss import GIoULoss
from mmcv.cnn.bricks.drop import Dropout

from aimet_torch.v2.nn import QuantizationMixin

QuantizationMixin.ignore(FocalLoss)
QuantizationMixin.ignore(L1Loss)
QuantizationMixin.ignore(GIoULoss)
QuantizationMixin.ignore(Dropout)

from quantization.registered_ops import QuantizedLinear

from evaluation.eval_dataset import extract_data
from evaluation.eval_dataset import build_eval_loader
from quantization.quantize_function import AimetTraceWrapper, create_quant_sim, calibration_forward_pass, move_to_device
from quantization.quant_techniques import maybe_run_bn_fold, maybe_run_cle, maybe_run_seq_mse
from ssr.projects.mmdet3d_plugin.SSR.model import load_default_model


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

@contextmanager
def debug_tensor_bool_mul():
    """
    Temporarily hook torch.mul and Tensor.__mul__ to print a stack trace
    whenever Tensor * bool happens. Useful for finding ONNX export failures
    like aten::mul(Tensor, bool).
    """
    orig_torch_mul = torch.mul
    orig_tensor_mul = torch.Tensor.__mul__

    def _debug_torch_mul(a, b, *args, **kwargs):
        a_is_tensor = isinstance(a, torch.Tensor)
        b_is_tensor = isinstance(b, torch.Tensor)

        if (a_is_tensor and isinstance(b, bool)) or (b_is_tensor and isinstance(a, bool)):
            print("\n[FOUND torch.mul Tensor-bool]")
            print("lhs:", type(a), getattr(a, "dtype", None))
            print("rhs:", type(b), getattr(b, "dtype", None) if b_is_tensor else type(b))
            traceback.print_stack(limit=20)

        return orig_torch_mul(a, b, *args, **kwargs)

    def _debug_tensor_mul(self, other):
        if isinstance(other, bool):
            print("\n[FOUND Tensor.__mul__ bool]")
            print("self dtype:", self.dtype)
            print("other:", other, type(other))
            traceback.print_stack(limit=20)
        return orig_tensor_mul(self, other)

    torch.mul = _debug_torch_mul
    torch.Tensor.__mul__ = _debug_tensor_mul

    try:
        yield
    finally:
        torch.mul = orig_torch_mul
        torch.Tensor.__mul__ = orig_tensor_mul

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
    cfg, dataset, data_loader = build_eval_loader(args.config, is_calib=True)

    print("Loading FP32 model...")
    model, _ = load_default_model(cfg, args.checkpoint, dataset, args.fuse_conv_bn, args.device)
    model = model.to(args.device).eval()

    first_batch = next(iter(data_loader))
    first_batch = extract_data(first_batch)
    prepared_batch = move_to_device(first_batch, torch.device(args.device))

    wrapped_model = AimetTraceWrapper(model=model).to(args.device).eval()
    wrapped_model.set_batch(prepared_batch)

    real_img = prepared_batch["img"][0]
    # real_img = prepared_batch["img"]

    if not torch.is_tensor(real_img):
        raise TypeError(f"Expected tensor, got {type(real_img)}")

    dummy_input = torch.zeros_like(real_img)

    torch.onnx.export(
        model,
        args=([dummy_input],),
        f="test/ssr.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17
    )

    maybe_run_cle(wrapped_model, dummy_input, args.enable_cle)
    maybe_run_bn_fold(wrapped_model, dummy_input, args.enable_bn_fold)

    skip_layer_names = []

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
        print("AIMET export completed successfully.")

    if args.export_onnx:
        export_dir = osp.join(args.work_dir, "onnx")
        onnx_path = osp.join(export_dir, f"{args.export_prefix}.onnx")
        print("Exporting quantized model to ONNX QDQ...")
        aimet_onnx.export(
            sim.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            export_int32_bias=False,
            prequantize_constants=True
        )
        print(f"Exported QDQ ONNX to: {onnx_path}")

    rank, _ = get_dist_info()
    if rank == 0:
        print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
