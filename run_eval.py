# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Phong Vu
# ---------------------------------------------
import os
import sys
import time
import os.path as osp

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
from mmcv import Config
print("mmcv loaded from:", mmcv.__file__)
import warnings
from mmcv import DictAction
from mmcv.runner import get_dist_info

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model
from ssr.projects.mmdet3d_plugin.SSR.model import load_default_model
from quantization.quantize_function import load_quantized_model

warnings.filterwarnings("ignore")

import platform
from mmcv.utils import Registry

if platform.system() != "Windows":
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry("Object sampler")

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    return None


def _flatten_result(obj):
    vals = []

    if obj is None:
        return vals

    arr = _to_numpy(obj)
    if arr is not None:
        vals.append(arr.reshape(-1))
        return vals

    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            vals.extend(_flatten_result(obj[k]))
        return vals

    if isinstance(obj, (list, tuple)):
        for item in obj:
            vals.extend(_flatten_result(item))
        return vals

    if isinstance(obj, (int, float, bool, np.number)):
        vals.append(np.array([obj], dtype=np.float32))
        return vals

    return vals


def compute_pcc(fp32_results, quant_results, eps=1e-12):
    fp32_flat = _flatten_result(fp32_results)
    quant_flat = _flatten_result(quant_results)

    if not fp32_flat or not quant_flat:
        return None, 0

    fp32_vec = np.concatenate(fp32_flat).astype(np.float64, copy=False)
    quant_vec = np.concatenate(quant_flat).astype(np.float64, copy=False)

    n = min(fp32_vec.size, quant_vec.size)
    if n == 0:
        return None, 0

    fp32_vec = fp32_vec[:n]
    quant_vec = quant_vec[:n]

    fp32_std = fp32_vec.std()
    quant_std = quant_vec.std()

    if fp32_std < eps or quant_std < eps:
        return None, n

    pcc = np.corrcoef(fp32_vec, quant_vec)[0, 1]
    return float(pcc), n

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    parser.add_argument("--config_path", help="test config file path")
    parser.add_argument("--fp32_weights", help="checkpoint file")
    parser.add_argument("--quant_weights", help="checkpoint file")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase the inference speed",
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, which depends on the dataset, e.g., "bbox", "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC',
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.',
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--graph_optimization_level",
        choices=["disable_all", "basic", "extended", "all"],
        default="disable_all",
        help="Quantization optimization level, only work with onnx format model",
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
    
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg, dataset, data_loader = build_eval_loader(args.config_path)

    rank, _ = get_dist_info()

    fp32_results = None
    quant_results = None

    # kwargs = {}
    # kwargs['jsonfile_prefix'] = osp.join('test', 'hihihahahuhu')

    # eval_kwargs = cfg.get('evaluation', {}).copy()
    # for key in [
    #     'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
    #     'rule'
    # ]:
    #     eval_kwargs.pop(key, None)
    # eval_kwargs.update(dict(metric=args.eval, **kwargs))


    if args.fp32_weights:
        print("Loading FP32 model...")
        fp32_model, _ = load_default_model(
            cfg,
            args.fp32_weights,
            dataset,
            args.fuse_conv_bn,
            args.device,
        )
        fp32_model.eval()

        print("Evaluating FP32 model...")
        fp32_results = evaluate_model(
            model_obj={
                "backend": "torch",
                "model": fp32_model,
                "session": None,
                "input_name": None,
                "output_names": None,
            },
            data_loader=data_loader,
            max_samples=args.max_samples,
        )

        if rank == 0:
            print("======================================================")
            print(dataset.evaluate(fp32_results, metric=args.eval))

    if args.quant_weights:
        quant_obj = load_quantized_model(
            quant_weights=args.quant_weights,
            device=args.device,
            graph_optimization_level=args.graph_optimization_level,
            provider=args.provider,
        )

        print("Evaluating quantized model...")
        quant_results = evaluate_model(
            model_obj=quant_obj,
            data_loader=data_loader,
            max_samples=args.max_samples,
        )

        if rank == 0:
            print("======================================================")
            print(dataset.evaluate(quant_results, metric=args.eval))

    if rank == 0 and fp32_results is not None and quant_results is not None:
        pcc, num_values = compute_pcc(fp32_results, quant_results)
        print("======================================================")
        if pcc is None:
            print(f"PCC could not be computed (num_values={num_values})")
        else:
            print(f"FP32 vs Quant PCC: {pcc:.8f} (num_values={num_values})")


if __name__ == "__main__":
    main()