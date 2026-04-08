import mmcv
import torch
import warnings

from evaluation.eval_dataset import extract_data

warnings.filterwarnings("ignore")

import numpy as np

from ssr.projects.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric


_PLANNING_METRIC = None


def compute_planner_metric_standalone(
    pred_ego_fut_trajs,
    gt_ego_fut_trajs,
    gt_agent_boxes,
    gt_agent_feats,
    gt_map_boxes,
    gt_map_labels,
    fut_valid_flag,
):
    """
    Copied from SSR.compute_planner_metric_stp3(), but standalone.
    """
    global _PLANNING_METRIC
    if _PLANNING_METRIC is None:
        _PLANNING_METRIC = PlanningMetric()

    metric_dict = {
        "plan_L2_1s": 0,
        "plan_L2_2s": 0,
        "plan_L2_3s": 0,
        "plan_obj_col_1s": 0,
        "plan_obj_col_2s": 0,
        "plan_obj_col_3s": 0,
        "plan_obj_box_col_1s": 0,
        "plan_obj_box_col_2s": 0,
        "plan_obj_box_col_3s": 0,
    }
    metric_dict["fut_valid_flag"] = fut_valid_flag

    future_second = 3
    assert pred_ego_fut_trajs.shape[0] == 1, "only support bs=1"

    segmentation, pedestrian, segmentation_plus = _PLANNING_METRIC.get_label(
        gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels
    )
    occupancy = torch.logical_or(segmentation, pedestrian)

    for i in range(future_second):
        if fut_valid_flag:
            cur_time = (i + 1) * 2

            traj_L2 = _PLANNING_METRIC.compute_L2(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
            )
            traj_L2_stp3 = _PLANNING_METRIC.compute_L2_stp3(
                pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                gt_ego_fut_trajs[0, :cur_time],
            )
            obj_coll, obj_box_coll = _PLANNING_METRIC.evaluate_coll(
                pred_ego_fut_trajs[:, :cur_time].detach(),
                gt_ego_fut_trajs[:, :cur_time],
                occupancy,
            )

            metric_dict[f"plan_L2_{i+1}s"] = traj_L2
            metric_dict[f"plan_obj_col_{i+1}s"] = obj_coll.mean().item()
            metric_dict[f"plan_obj_box_col_{i+1}s"] = obj_box_coll.mean().item()

            metric_dict[f"plan_L2_stp3_{i+1}s"] = traj_L2_stp3
            metric_dict[f"plan_obj_col_stp3_{i+1}s"] = obj_coll[-1].item()
            metric_dict[f"plan_obj_box_col_stp3_{i+1}s"] = obj_box_coll[-1].item()
        else:
            metric_dict[f"plan_L2_{i+1}s"] = 0.0
            metric_dict[f"plan_obj_col_{i+1}s"] = 0.0
            metric_dict[f"plan_obj_box_col_{i+1}s"] = 0.0
            metric_dict[f"plan_L2_stp3_{i+1}s"] = 0.0

    return metric_dict

def run_onnx_ssr(model_obj, data):
    session = model_obj["session"]
    input_name = model_obj["input_name"]

    if input_name is None:
        raise ValueError(
            "ONNX model has no input name. "
            "The exported wrapped graph likely does not depend on the formal input tensor."
        )

    img = data["img"]
    if isinstance(img, list):
        assert len(img) == 1, f"Unexpected img list length: {len(img)}"
        img = img[0]

    if not torch.is_tensor(img):
        raise TypeError(f"Expected torch.Tensor for ONNX input, got {type(img)}")

    img_np = img.detach().cpu().numpy()

    if img_np.ndim == 6:
        if img_np.shape[1] == 1:
            img_np = img_np[:, 0]
        else:
            raise ValueError(f"Unexpected 6D input shape for ONNX: {img_np.shape}")

    if img_np.ndim == 5:
        b, n, c, h, w = img_np.shape
        img_np = img_np.reshape(b * n, c, h, w)

    if img_np.ndim != 4:
        raise ValueError(f"ONNX expects rank-4 input, got shape {img_np.shape}")

    outputs = session.run(None, {input_name: img_np})
    output_names = [o.name for o in session.get_outputs()]

    outs = {
        name: torch.from_numpy(arr)
        for name, arr in zip(output_names, outputs)
    }

    # Recover semantic tensors by shape if export names are generic
    if "ego_fut_preds" not in outs:
        for k, v in outs.items():
            if v.ndim == 4 and tuple(v.shape[-3:]) == (3, 6, 2):
                outs["ego_fut_preds"] = v
                break

    if "bev_embed" not in outs:
        for k, v in outs.items():
            if v.ndim == 3 and v.shape[-1] == 256:
                outs["bev_embed"] = v
                break

    if "ego_fut_preds" not in outs:
        raise KeyError(f"Could not find ego_fut_preds in ONNX outputs: {list(outs.keys())}")

    return outs

def unwrap_single(x, name="value"):
    while isinstance(x, list):
        if len(x) != 1:
            raise ValueError(f"{name} expected single-item list nesting, got len={len(x)}")
        x = x[0]
    return x

def build_ssr_result_from_onnx_outs(outs, data):
    ego_fut_cmd = unwrap_single(data["ego_fut_cmd"], "ego_fut_cmd")
    ego_fut_trajs = unwrap_single(data["ego_fut_trajs"], "ego_fut_trajs")
    fut_valid_flag = unwrap_single(data["fut_valid_flag"], "fut_valid_flag")
    gt_bboxes_3d = data["gt_bboxes_3d"]
    map_gt_bboxes_3d = data["map_gt_bboxes_3d"]
    map_gt_labels_3d = data["map_gt_labels_3d"]
    gt_attr_labels = data["gt_attr_labels"]

    bbox_results = []
    for i in range(len(outs["ego_fut_preds"])):
        bbox_results.append({
            "ego_fut_preds": outs["ego_fut_preds"][i].cpu(),
            "ego_fut_cmd": ego_fut_cmd.cpu(),
        })

    assert len(bbox_results) == 1, "only support batch_size=1 now"

    with torch.no_grad():
        gt_bbox = gt_bboxes_3d[0][0]
        gt_map_bbox = map_gt_bboxes_3d[0]
        gt_map_label = map_gt_labels_3d[0].to("cpu")
        gt_attr_label = gt_attr_labels[0][0].to("cpu")
        fut_valid = bool(fut_valid_flag[0])

        ego_fut_preds = bbox_results[0]["ego_fut_preds"]
        gt_ego_fut_trajs = ego_fut_trajs[0, 0]
        cmd = ego_fut_cmd[0, 0, 0]
        cmd_idx = torch.nonzero(cmd)[0, 0]

        ego_fut_pred = ego_fut_preds[cmd_idx]
        ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
        gt_ego_fut_trajs = gt_ego_fut_trajs.cumsum(dim=-2)

        metric_dict = compute_planner_metric_standalone(
            pred_ego_fut_trajs=ego_fut_pred[None],
            gt_ego_fut_trajs=gt_ego_fut_trajs[None],
            gt_agent_boxes=gt_bbox,
            gt_agent_feats=gt_attr_label.unsqueeze(0),
            gt_map_boxes=gt_map_bbox,
            gt_map_labels=gt_map_label,
            fut_valid_flag=fut_valid,
        )

    return [{
        "pts_bbox": bbox_results[0],
        "metric_results": metric_dict,
    }]

def get_model_result(model_obj, data):
    backend = model_obj["backend"]

    if backend == "torch":
        model = model_obj["model"]
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        return result

    elif backend == "onnx":
        outs = run_onnx_ssr(model_obj, data)
        result = build_ssr_result_from_onnx_outs(outs, data)
        return result

    raise ValueError(f"Unsupported backend: {backend}")

def maybe_dump_heatmaps(model_obj):
    if model_obj["backend"] != "torch":
        return

    model = model_obj["model"]

    if not hasattr(model, "pts_bbox_head"):
        return

    try:
        heatmaps_list = model.pts_bbox_head.transformer.encoder.layers[0].attentions[1]._heatmaps_list
    except Exception:
        return

    if not heatmaps_list:
        return

    from pathlib import Path
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    output_dir = Path("logs/run_heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    final_heatmap = None
    for cam_idx, heatmap in enumerate(heatmaps_list):
        if isinstance(heatmap, torch.Tensor):
            heatmap_np = heatmap.detach().cpu().float().numpy()
        else:
            heatmap_np = np.asarray(heatmap, dtype=np.float32)

        heatmap_np = np.clip(
            heatmap_np,
            a_min=0.0,
            a_max=None,
        ).astype(np.int32, copy=False)

        if heatmap_np.ndim != 2 or heatmap_np.shape != (100, 100):
            continue

        if final_heatmap is None:
            final_heatmap = heatmap_np.copy()
        else:
            final_heatmap += heatmap_np

        save_path = output_dir / f"camera_{cam_idx}_heatmap.png"
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(
            heatmap_np,
            annot=True,
            annot_kws={"size": 3},
            square=True,
            cbar=True,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if final_heatmap is not None:
        final_path = output_dir / "final_heatmap.png"
        fig, ax = plt.subplots(figsize=(11, 11))
        sns.heatmap(
            final_heatmap,
            annot=True,
            annot_kws={"size": 3},
            square=True,
            cbar=True,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(final_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def evaluate_model(
    model_obj,
    data_loader,
    max_samples=20,
):
    if isinstance(model_obj, dict):
        backend = model_obj.get("backend", "torch")
        model = model_obj.get("model", None)
        normalized_model_obj = model_obj
    else:
        backend = getattr(model_obj, "backend", "torch")
        model = model_obj
        normalized_model_obj = {
            "backend": backend,
            "model": model,
        }

    if backend == "torch" and model is not None:
        model.eval()

    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    try:
        for i, data in enumerate(data_loader):
            if max_samples is not None and i >= max_samples:
                break

            data = extract_data(data)
            result = get_model_result(normalized_model_obj, data)

            results.extend(result)

            batch_size = len(result)
            for _ in range(batch_size):
                prog_bar.update()

    except KeyboardInterrupt:
        print("Keyboard interrupt, exiting...")

    maybe_dump_heatmaps(normalized_model_obj)
    return results