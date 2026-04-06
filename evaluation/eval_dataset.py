import os

from mmcv import Config
from mmcv.utils import build_from_cfg
from mmdet.datasets import DATASETS
from utils.config import prepare_cfg

from ssr.projects.mmdet3d_plugin.datasets.builder import build_dataloader

def build_dataset(cfg, default_args=None):
    """Build a dataset from an MMDetection-style dataset config."""
    return build_from_cfg(cfg, DATASETS, default_args)


# def extract_data(data):
#     """Extract tensor payloads from MMCV DataContainer fields."""
#     data["img_metas"] = data["img_metas"][0].data
#     data["gt_bboxes_3d"] = data["gt_bboxes_3d"][0].data
#     data["gt_labels_3d"] = data["gt_labels_3d"][0].data
#     data["img"] = data["img"][0].data
#     data["ego_his_trajs"] = data["ego_his_trajs"][0].data
#     data["ego_fut_trajs"] = data["ego_fut_trajs"][0].data
#     data["ego_fut_cmd"] = data["ego_fut_cmd"][0].data
#     data["ego_lcf_feat"] = data["ego_lcf_feat"][0].data
#     data["gt_attr_labels"] = data["gt_attr_labels"][0].data
#     data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
#     data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
#     return data

def unwrap_datacontainer(x):
    while hasattr(x, "data"):
        x = x.data
    return x

def unwrap_singleton_list(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x

def extract_data(data):
    data = dict(data)

    keys = [
        "img_metas",
        "gt_bboxes_3d",
        "gt_labels_3d",
        "img",
        "ego_his_trajs",
        "ego_fut_trajs",
        "ego_fut_cmd",
        "ego_lcf_feat",
        "gt_attr_labels",
        "map_gt_labels_3d",
        "map_gt_bboxes_3d",
    ]

    for k in keys:
        if k in data:
            data[k] = unwrap_datacontainer(data[k])
            data[k] = unwrap_singleton_list(data[k])

    return data

def build_eval_loader(
    config_path,
    cfg_options=None,
    distributed=False,
    shuffle=False,
):
    """Create dataset and dataloader for test/inference from a config file.

    Args:
        config_path (str): Path to the MMCV config file.
        cfg_options (dict | None): Optional config overrides.
        distributed (bool): Whether to build a distributed dataloader.
        shuffle (bool): Whether to shuffle the dataloader.

    Returns:
        tuple: (cfg, dataset, data_loader)
    """
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)

    cfg.model.pretrained = None
    cfg, samples_per_gpu = prepare_cfg(cfg)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=shuffle,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    return cfg, dataset, data_loader
