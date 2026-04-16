# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Phong Vu
# ---------------------------------------------
import os
import sys
import time
import os.path as osp
import argparse
from loguru import logger 
import warnings
import pathlib

py_deps = os.environ.get("PY_DEPS_DIR")
if py_deps:
    if py_deps in sys.path:
        sys.path.remove(py_deps)
    sys.path.insert(0, py_deps)

sys.path.append("")

print("PY_DEPS_DIR =", py_deps)
print("sys.path[:6] =", sys.path[:6])

import numpy as np
import torch
import mmcv
from mmcv import Config
print("mmcv loaded from:", mmcv.__file__)
import warnings
from mmcv import DictAction
from mmcv.runner import get_dist_info

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

import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info

from ssr.projects.mmdet3d_plugin.datasets import VADCustomNuScenesDataset
from ssr.projects.mmdet3d_plugin.SSR.runner.runner import SSRRunner

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1])) 

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")

def get_output_path(out_dir='generated/video'):
    current_path = os.getcwd()
    current_path = current_path.rsplit('/', 1)[0] + '/'
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(current_path, out_dir)

    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(out_dir, f"realtime_{ts}.mp4")
    return video_path

def run_inference(runner, data_loader, **kwargs):
    results = []
    dataset = data_loader.dataset

    try:
        while True:
            prog_bar = mmcv.ProgressBar(len(dataset))
            for i, data in enumerate(data_loader):
                result = runner(data, sample_idx=i, mode="fp32", **kwargs)
                quant_result = runner(data, sample_idx=i, mode="quant", **kwargs)

                if kwargs.get("visualize", False):
                    runner._post_process_compare(result, quant_result, data, sample_idx=i, **kwargs)

                if not kwargs.get("repeat", False): 
                    results.extend(result)

                prog_bar.update()

            if not kwargs.get("repeat", False): 
                break

        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
    except (KeyboardInterrupt, SystemExit):
        if kwargs.get("visualize", False):
            runner.close_visualizer(**kwargs)
        logger.info("KeyboardInterrupt or SystemExit detected, exiting...")

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="MMDet test (and eval) a model")
    # Configs
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Choose device",
    )

    parser.add_argument("--config_path", help="Torch model config file", default=None)
    parser.add_argument("--config", help="TTNN model config file", default=None)

    # Checkpoints
    parser.add_argument("--fp32_checkpoint_path", help="Torch checkpoint file", default=None)
    parser.add_argument("--quant_checkpoint_path", help="Quantized checkpoint file")
    parser.add_argument("--encodings_path", help="Quantized encodings file")

    parser.add_argument("--enable_bn_fold", action="store_true", help="apply AIMET BN fold")

    parser.add_argument("--visualize", action="store_true",
                    help="enable realtime visualization")
    parser.add_argument("--double_cq", action="store_true",
                    help="Hihi haha huhu")
    parser.add_argument("--bev_map", action="store_true",
                        help="enable BEV map during realtime visualization", default=False)
    parser.add_argument("--realtime", action="store_true",
                        help="show inference directly or save it as video")
    parser.add_argument("--max_samples", type=int, default=323)
    # Others
    parser.add_argument("--out", help="output result file in pickle format")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--repeat", action="store_true",
                        help="repeat demo until user interrupts")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    cfg, dataset, data_loader = build_eval_loader(args.config)

    runner = SSRRunner(
        device=args.device,
        fp32_checkpoint_path=args.fp32_checkpoint_path,
        quant_checkpoint_path=args.quant_checkpoint_path,
        encodings_path=args.encodings_path,
        config_path=args.config_path,
        config=args.config,
        double_cq=args.double_cq,
        dataset=data_loader.dataset,
        enable_bn_fold=args.enable_bn_fold,
    )

    outputs = run_inference(
        runner, 
        data_loader,
        visualize=args.visualize, 
        realtime=args.realtime,
        use_bev=args.bev_map,
        repeat=args.repeat,
    )

    # Save results
    tmp = {}
    tmp["bbox_results"] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0 and args.out:
        logger.info(f"\nwriting results to {args.out}")
        if isinstance(outputs, list):
            mmcv.dump(outputs, args.out)
        else:
            mmcv.dump(outputs["bbox_results"], args.out)


if __name__ == "__main__":
    main()