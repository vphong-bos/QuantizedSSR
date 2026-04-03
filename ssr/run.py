# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
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
print("mmcv loaded from:", mmcv.__file__)
import torch
import warnings
import numpy as np
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model

from ssr.projects.mmdet3d_plugin.datasets.builder import build_dataloader
from ssr.projects.mmdet3d_plugin.SSR.utils.builder import build_model
from ssr.projects.mmdet3d_plugin.datasets import VADCustomNuScenesDataset

from quantization.quantize_function import AimetTraceWrapper, aimet_forward_fn
from aimet_torch.v2 import quantsim

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp

warnings.filterwarnings("ignore")

import platform
from mmcv.utils import Registry, build_from_cfg
from mmdet.datasets import DATASETS

if platform.system() != 'Windows':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))


def extract_data_from_container(data):
    """Extract data from DataContainer."""
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


OBJECTSAMPLERS = Registry('Object sampler')


def build_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DATASETS, default_args)


def get_onnx_opt_level(opt_level_str):
    import onnxruntime as ort

    mapping = {
        'disable': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        'basic': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        'extended': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        'all': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    return mapping[opt_level_str]


def load_onnx_session(onnx_path, provider='CPUExecutionProvider', opt_level='basic'):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = get_onnx_opt_level(opt_level)

    session = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=[provider],
    )

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print('[ONNX] loaded from:', onnx_path)
    print('[ONNX] provider   :', provider)
    print('[ONNX] opt level  :', opt_level)
    print('[ONNX] input      :', input_name)
    print('[ONNX] outputs    :', output_names)

    return {
        "backend": "onnx",
        "session": session,
        "input_name": input_name,
        "output_names": output_names,
        "model": None,
    }


def build_normal_model(cfg, checkpoint_path=None, fuse_conv_bn_flag=False):
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)

    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    if fuse_conv_bn_flag:
        model = fuse_conv_bn(model)

    return model, checkpoint


def load_aimet_quantized_model(
    quant_weights,
    device,
    provider="CPUExecutionProvider",
    opt_level="basic",
):
    print("Loading quantized model...")

    ext = os.path.splitext(quant_weights)[1].lower()

    # =========================
    # Case 1: ONNX QDQ model
    # =========================
    if ext == ".onnx":
        print("Detected ONNX model")
        return load_onnx_session(
            quant_weights,
            provider=provider,
            opt_level=opt_level,
        )

    # =========================
    # Case 2: AIMET checkpoint
    # =========================
    print("Detected AIMET checkpoint")

    sim = quantsim.load_checkpoint(quant_weights)
    sim.model.to(device).eval()

    return {
        "backend": "torch",
        "model": sim.model,
        "session": None,
        "input_name": None,
        "output_names": None,
    }


def apply_checkpoint_metadata(model, checkpoint, dataset):
    if checkpoint is None:
        return

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE


def wrap_with_aimet_trace(model):
    return AimetTraceWrapper(model, aimet_forward_fn)


def run_normal_inference(model, data):
    with torch.no_grad():
        return model(return_loss=False, rescale=True, **data)


def run_aimet_inference(model, data):
    with torch.no_grad():
        return model(return_loss=False, rescale=True, **data)


def run_onnx_inference(session, input_name, data):
    img = data["img"]
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()
    else:
        img_np = np.asarray(img)

    ort_inputs = {input_name: img_np}
    ort_outputs = session.run(None, ort_inputs)

    # TODO: convert ort_outputs to the same format as torch inference output
    return ort_outputs


def run_inference(model_obj, mode, data):
    if mode == 'normal':
        return run_normal_inference(model_obj["model"], data)
    if mode == 'aimet_trace':
        return run_aimet_inference(model_obj["model"], data)
    if mode == 'aimet_quant':
        if model_obj["backend"] == "onnx":
            return run_onnx_inference(model_obj["session"], model_obj["input_name"], data)
        return run_aimet_inference(model_obj["model"], data)
    raise ValueError(f'Unsupported inference mode: {mode}')


def maybe_dump_heatmaps(model_obj, mode):
    if mode not in ['normal', 'aimet_trace']:
        return

    model = model_obj["model"]
    if not hasattr(model, 'pts_bbox_head'):
        return

    try:
        heatmaps_list = model.pts_bbox_head.transformer.encoder.layers[0].attentions[1]._heatmaps_list
    except Exception:
        return

    if not heatmaps_list:
        return

    from pathlib import Path
    import seaborn as sns
    import matplotlib.pyplot as plt

    output_dir = Path('logs/run_heatmaps')
    output_dir.mkdir(parents=True, exist_ok=True)

    final_heatmap = None
    for cam_idx, heatmap in enumerate(heatmaps_list):
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().cpu().float().numpy()
        else:
            heatmap_np = np.asarray(heatmap, dtype=np.float32)

        heatmap_np = np.clip(heatmap_np, a_min=0.0, a_max=None).astype(np.int32, copy=False)
        if heatmap_np.ndim != 2 or heatmap_np.shape != (100, 100):
            continue

        if final_heatmap is None:
            final_heatmap = heatmap_np.copy()
        else:
            final_heatmap += heatmap_np

        save_path = output_dir / f'camera_{cam_idx}_heatmap.png'
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(
            heatmap_np,
            annot=True,
            annot_kws={'size': 3},
            square=True,
            cbar=True,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    if final_heatmap is not None:
        final_path = output_dir / 'final_heatmap.png'
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(
            final_heatmap,
            annot=True,
            annot_kws={'size': 3},
            square=True,
            cbar=True,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(final_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def single_gpu_test(model_obj,
                    data_loader,
                    max_samples=20,
                    mode='normal'):
    if model_obj["backend"] == "torch" and hasattr(model_obj["model"], 'eval'):
        model_obj["model"].eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    try:
        for i, data in enumerate(data_loader):
            if max_samples is not None and i >= max_samples:
                break

            data = extract_data_from_container(data)
            result = run_inference(
                model_obj=model_obj,
                mode=mode,
                data=data,
            )

            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()

    except KeyboardInterrupt:
        print('Keyboard interrupt, exiting...')

    maybe_dump_heatmaps(model_obj, mode)
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--quant-weights', help='aimet checkpoint or onnx quantized model')
    parser.add_argument(
        '--model-type',
        type=str,
        default='normal',
        choices=['normal', 'aimet_trace', 'aimet_quant'],
        help='inference backend type')
    parser.add_argument('--json_dir', help='json parent dir name file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
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
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--return_model', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument(
        '--onnx-provider',
        type=str,
        default='CPUExecutionProvider',
        choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
        help='ONNX Runtime execution provider')
    parser.add_argument(
        '--onnx-opt-level',
        type=str,
        default='basic',
        choices=['disable', 'basic', 'extended', 'all'],
        help='ONNX Runtime graph optimization level')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options

    if args.model_type == 'normal':
        if args.checkpoint is None:
            raise ValueError('--model-type normal requires --checkpoint')
    else:
        if args.quant_weights is None:
            raise ValueError(f'--model-type {args.model_type} requires --quant-weights')

    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                importlib.import_module(_module_path)
            else:
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                importlib.import_module(_module_path)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None

    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    distributed = False

    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    mode = args.model_type

    if mode == 'normal':
        model, checkpoint = build_normal_model(
            cfg=cfg,
            checkpoint_path=args.checkpoint,
            fuse_conv_bn_flag=args.fuse_conv_bn,
        )
        apply_checkpoint_metadata(model, checkpoint, dataset)
        model_obj = {
            "backend": "torch",
            "model": model,
            "session": None,
            "input_name": None,
            "output_names": None,
        }

    elif mode == 'aimet_trace':
        model, checkpoint = build_normal_model(
            cfg=cfg,
            checkpoint_path=args.quant_weights,
            fuse_conv_bn_flag=args.fuse_conv_bn,
        )
        apply_checkpoint_metadata(model, checkpoint, dataset)
        model = wrap_with_aimet_trace(model)
        model_obj = {
            "backend": "torch",
            "model": model,
            "session": None,
            "input_name": None,
            "output_names": None,
        }

    elif mode == 'aimet_quant':
        model_obj = load_aimet_quantized_model(
            quant_weights=args.quant_weights,
            device=args.device,
            provider=args.onnx_provider,
            opt_level=args.onnx_opt_level,
        )
    else:
        raise ValueError(f'Unsupported model type: {mode}')

    if args.return_model:
        return model_obj, dataset

    outputs = single_gpu_test(
        model_obj,
        data_loader,
        args.max_samples,
        mode=mode,
    )

    outputs = {'bbox_results': outputs}
    rank, _ = get_dist_info()

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs['bbox_results'], args.out)

        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join(
            'test',
            args.config.split('/')[-1].split('.')[-2],
            time.ctime().replace(' ', '_').replace(':', '_')
        )

        if args.format_only:
            dataset.format_results(outputs['bbox_results'], **kwargs)

        if args.eval:
            print("======================================================")
            print(dataset.evaluate(outputs['bbox_results'], metric=args.eval))


if __name__ == '__main__':
    main()