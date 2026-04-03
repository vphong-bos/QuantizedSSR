# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Phong Vu
# ---------------------------------------------
import os
import sys

py_deps = os.environ.get("PY_DEPS_DIR")
if py_deps:
    if py_deps in sys.path:
        sys.path.remove(py_deps)
    sys.path.insert(0, py_deps)

sys.path.append('')

print("PY_DEPS_DIR =", py_deps)
print("sys.path[:6] =", sys.path[:6])

import argparse
import mmcv
from mmcv import Config
print("mmcv loaded from:", mmcv.__file__)
import warnings
from mmcv import DictAction
from mmcv.runner import (get_dist_info)

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model
from ssr.projects.mmdet3d_plugin.SSR.model import build_model
from quantization.quantize_function import load_quantized_model

from utils.config import import_plugin_modules, prepare_cfg

import warnings
warnings.filterwarnings("ignore")


import platform
from mmcv.utils import Registry

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--fp32_weights', help='checkpoint file')
    parser.add_argument('--quant_weights', help='checkpoint file')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--graph_optimization_level',
        choices=["disable_all", "basic", "extended", "all"],
        help='Quantization optimization level, only work with onnx format model')
    parser.add_argument(
        '--device',
        choices=["cpu", "cuda"],
        default="cpu"
        help='Choose device')
    parser.add_argument(
        '--provider',
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        default="CPUExecutionProvider",
        help='Choose onnx provider')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg, dataset, data_loader = build_eval_loader(args.config_path, args.cfg_options)

    rank, _ = get_dist_info()

    if args.fp32_weights:
        print("Loading FP32 model...")
        fp32_model, _ = build_model(cfg, args.fp32_weights, dataset, args.fuse_conv_bn, args.device)
        print("Evaluating FP32 model...")
        fp32_results = evaluate_model(
            model_obj={
                "backend": "torch",
                "model": fp32_model,
                "session": None,
                "input_name": None,
                "output_names": None,
            }, 
            loader=data_loader, 
            max_samples=args.max_samples,
        )

        print(fp32_results)
        tmp = {}
        tmp['bbox_results'] = fp32_results
        outputs = tmp
        if rank == 0:
            print("======================================================")
            print(dataset.evaluate(fp32_results['bbox_results'], metric=args.eval))

    if args.quant_weights:
        quant_obj = load_quantized_model(
            quant_weights=args.quant_weights,
            device=args.device,
            graph_optimization_level=args.graph_optimization_level,
            provider=args.provider,
        )
        
        quant_results = evaluate_model(
            model_obj=quant_obj,
            loader=data_loader,
            max_samples=args.max_samples,
        )

        print(quant_results)
        tmp = {}
        tmp['bbox_results'] = quant_results
        quant_results = tmp
        if rank == 0:
            print("======================================================")
            print(dataset.evaluate(quant_results['bbox_results'], metric=args.eval))

if __name__ == '__main__':
    main()
    
    
