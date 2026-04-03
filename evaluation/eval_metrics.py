import mmcv
import torch
import warnings

from evaluation.eval_dataset import extract_data

warnings.filterwarnings("ignore")

def get_model_result(model_obj, data):
    backend = model_obj["backend"]

    if backend == "torch":
        model = model_obj["model"]
        with torch.no_grad():
            if hasattr(model, "set_batch"):
                model.set_batch(data)
                img = data["img"]
                if isinstance(img, list):
                    assert len(img) == 1, f"Unexpected img list length: {len(img)}"
                    img = img[0]
                result = model(img)
            else:
                result = model(return_loss=False, rescale=True, **data)
        return result

    elif backend == "onnx":
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

        # adapt wrapped-model export input shape
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

        result = session.run(None, {input_name: img_np})
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