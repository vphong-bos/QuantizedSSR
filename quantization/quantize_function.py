import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime_extensions import (
    PyCustomOpDef,
    enable_py_op,
    get_library_path,
    onnx_op,
)

from aimet_common.defs import QuantScheme
from aimet_torch import quantsim
from aimet_torch.v2.nn import QuantizationMixin
from aimet_torch.v2.quantsim import QuantizationSimModel

from evaluation.eval_dataset import extract_data, build_eval_loader
from ssr.projects.mmdet3d_plugin.SSR.utils.builder import build_model


def unwrap_datacontainer(x: Any) -> Any:
    while hasattr(x, "data"):
        x = x.data
    return x


def move_to_device(obj: Any, device: torch.device, non_blocking: bool = True) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return [move_to_device(x, device, non_blocking) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device, non_blocking) for x in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    return obj


def prepare_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return move_to_device(extract_data(batch), device)


def extract_single_img(batch: Dict[str, Any]) -> torch.Tensor:
    img = batch["img"]
    if isinstance(img, list):
        if len(img) != 1:
            raise ValueError(f"Unexpected img list length: {len(img)}")
        img = img[0]
    if not torch.is_tensor(img):
        raise TypeError(f"Expected img to be a tensor, got {type(img)}")
    return img


def ensure_trace_img_shape(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 4:
        return img.unsqueeze(0)
    if img.ndim != 5:
        raise ValueError(f"Unexpected img shape in trace mode: {tuple(img.shape)}")
    return img


class AimetTraceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.runtime_batch: Optional[Dict[str, Any]] = None

    def set_batch(self, batch: Dict[str, Any]) -> None:
        self.runtime_batch = batch

    def forward(self, img: Optional[torch.Tensor] = None, **kwargs):
        if kwargs:
            return self.model(img=img, **kwargs)

        if self.runtime_batch is None:
            raise RuntimeError("runtime_batch must be set before tracing")
        if img is None:
            raise ValueError("img must not be None in trace mode")

        img = ensure_trace_img_shape(img)
        batch = self.runtime_batch

        return self.model.forward_trace(
            img_metas=batch["img_metas"],
            img=[img],
            ego_fut_cmd=batch["ego_fut_cmd"],
            ego_his_trajs=batch["ego_his_trajs"],
            ego_lcf_feat=batch["ego_lcf_feat"],
        )


def aimet_forward_fn(model: nn.Module, inputs: Any):
    if isinstance(inputs, dict):
        img = inputs["img"]
    elif isinstance(inputs, (list, tuple)):
        img = inputs[0]
    else:
        img = inputs

    if isinstance(img, list):
        if len(img) != 1:
            raise ValueError(f"Unexpected img list length: {len(img)}")
        img = img[0]

    img = img.to(next(model.parameters()).device)
    return model(img)


@torch.no_grad()
def run_model_on_batch(
    model: AimetTraceWrapper,
    batch: Dict[str, Any],
    device: torch.device,
):
    batch = prepare_batch(batch, device)
    model.set_batch(batch)
    img = extract_single_img(batch)
    return model(img)


def calibration_forward_pass(model, forward_pass_args) -> None:
    data_loader, device, calib_batches, calib_max_samples = forward_pass_args

    model.eval()
    seen = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = prepare_batch(data, device)
            model(return_loss=False, rescale=True, **data)

            batch_img = extract_single_img(data)
            batch_size = batch_img.shape[0]
            seen += batch_size

            if calib_batches is not None and i + 1 >= calib_batches:
                break
            if calib_max_samples is not None and seen >= calib_max_samples:
                break


def get_onnx_graph_optimization_level(level):
    """
    Map a user-friendly string or enum-like value to an ONNX Runtime
    graph optimization level.
    """
    if not isinstance(level, str):
        return level

    mapping = {
        "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    key = level.strip().lower()
    if key not in mapping:
        raise ValueError(
            f"Unsupported graph_optimization_level: {level}. "
            f"Choose from {list(mapping.keys())}"
        )
    return mapping[key]


def get_named_modules_to_ignore(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    exclude_keywords = [
        "positional_encoding",
        "embedding",
        "norm",
        "reference_points",
        "map_reference_points",
        "attention_weights",
    ]

    ignored: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if any(keyword in name for keyword in exclude_keywords):
            ignored.append((name, module))
    return ignored


def apply_quantmixin_ignore(
    modules_to_ignore: List[Tuple[str, nn.Module]],
) -> None:
    if not modules_to_ignore:
        return

    modules_only = []
    for item in modules_to_ignore:
        if isinstance(item, tuple):
            name, module = item
        else:
            name, module = None, item

        if module is None:
            continue

        modules_only.append(module)
        if name is not None:
            print(f"[QuantizationMixin.ignore] {name}")

    QuantizationMixin.ignore(modules_only)


def create_quant_sim(
    model: nn.Module,
    device: torch.device,
    dummy_input: torch.Tensor,
    quant_scheme: str,
    default_output_bw: int,
    default_param_bw: int,
    config_path: Optional[str] = None,
    skip_layer_names: Optional[List[str]] = None,
) -> QuantizationSimModel:
    scheme_map = {
        "tf": QuantScheme.post_training_tf,
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    }
    selected_scheme = scheme_map.get(
        quant_scheme,
        QuantScheme.post_training_tf_enhanced,
    )

    sim = QuantizationSimModel(
        model=model.to(device).eval(),
        dummy_input=dummy_input,
        quant_scheme=selected_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file=config_path,
        in_place=False,
    )

    name_to_module = dict(sim.model.named_modules())
    layers_to_exclude = []
    missing = []

    for name in skip_layer_names or []:
        if name in name_to_module:
            layers_to_exclude.append(name_to_module[name])
            print(f"[EXCLUDE LAYER] {name}")
        else:
            missing.append(name)

    if missing:
        print("Missing layer names:")
        for name in missing:
            print(f"  {name}")

    if layers_to_exclude:
        sim.exclude_layers_from_quantization(layers_to_exclude)

    return sim


def _unnormalize(coord: float, size: int, align_corners: bool) -> float:
    if align_corners:
        return ((coord + 1.0) * (size - 1)) / 2.0
    return ((coord + 1.0) * size - 1.0) / 2.0


@onnx_op(
    op_type="GridSampleBilinearZerosAC0",
    domain="ai.onnx.contrib",
    inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float],
    outputs=[PyCustomOpDef.dt_float],
)
def grid_sample_bilinear_zeros_ac0(x, grid):
    x = np.asarray(x, dtype=np.float32)
    grid = np.asarray(grid, dtype=np.float32)

    if x.ndim != 4:
        raise ValueError(f"Expected x.ndim == 4, got {x.ndim}")
    if grid.ndim != 4 or grid.shape[-1] != 2:
        raise ValueError(
            f"Expected grid shape [N, H_out, W_out, 2], got {grid.shape}"
        )

    n, c, h, w = x.shape
    ng, h_out, w_out, _ = grid.shape
    if ng != n:
        raise ValueError(f"Batch mismatch: x batch {n}, grid batch {ng}")

    y = np.zeros((n, c, h_out, w_out), dtype=np.float32)

    for batch_idx in range(n):
        for out_h in range(h_out):
            for out_w in range(w_out):
                gx = grid[batch_idx, out_h, out_w, 0]
                gy = grid[batch_idx, out_h, out_w, 1]

                ix = _unnormalize(gx, w, align_corners=False)
                iy = _unnormalize(gy, h, align_corners=False)

                x0 = int(np.floor(ix))
                x1 = x0 + 1
                y0 = int(np.floor(iy))
                y1 = y0 + 1

                wa = (x1 - ix) * (y1 - iy)
                wb = (ix - x0) * (y1 - iy)
                wc = (x1 - ix) * (iy - y0)
                wd = (ix - x0) * (iy - y0)

                if 0 <= x0 < w and 0 <= y0 < h:
                    y[batch_idx, :, out_h, out_w] += wa * x[batch_idx, :, y0, x0]
                if 0 <= x1 < w and 0 <= y0 < h:
                    y[batch_idx, :, out_h, out_w] += wb * x[batch_idx, :, y0, x1]
                if 0 <= x0 < w and 0 <= y1 < h:
                    y[batch_idx, :, out_h, out_w] += wc * x[batch_idx, :, y1, x0]
                if 0 <= x1 < w and 0 <= y1 < h:
                    y[batch_idx, :, out_h, out_w] += wd * x[batch_idx, :, y1, x1]

    return y


def maybe_run_bn_fold(
    wrapped_model: AimetTraceWrapper,
    dummy_input: torch.Tensor,
    enabled: bool,
) -> None:
    if not enabled:
        print("BN folding disabled")
        return

    print("Applying AIMET batch norm folding...")

    from aimet_torch.batch_norm_fold import fold_all_batch_norms

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


def log_uninitialized_quantizers(sim: QuantizationSimModel) -> None:
    for name, module in sim.model.named_modules():
        if hasattr(module, "input_quantizers"):
            for i, quantizer in enumerate(module.input_quantizers):
                if quantizer is None:
                    continue
                try:
                    quantizer.get_scale()
                except Exception:
                    print(f"UNINIT input quantizer: {name}.input_quantizers[{i}]")

        if hasattr(module, "output_quantizers"):
            for i, quantizer in enumerate(module.output_quantizers):
                if quantizer is None:
                    continue
                try:
                    quantizer.get_scale()
                except Exception:
                    print(f"UNINIT output quantizer: {name}.output_quantizers[{i}]")

        if hasattr(module, "param_quantizers"):
            for param_name, quantizer in module.param_quantizers.items():
                if quantizer is None:
                    continue
                try:
                    quantizer.get_scale()
                except Exception:
                    print(
                        f"UNINIT param quantizer: "
                        f"{name}.param_quantizers['{param_name}']"
                    )


def build_onnx_session(
    quant_weights: str,
    provider: str,
    graph_optimization_level: str,
) -> Dict[str, Any]:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = get_onnx_graph_optimization_level(
        graph_optimization_level
    )

    enable_py_op(True)
    session_options.register_custom_ops_library(get_library_path())

    session = ort.InferenceSession(
        quant_weights,
        sess_options=session_options,
        providers=[provider],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    return {
        "backend": "onnx",
        "session": session,
        "input_name": inputs[0].name if inputs else None,
        "output_names": [output.name for output in outputs],
        "model": None,
        "sim": None,
        "graph_optimization_level": graph_optimization_level,
        "encoding_path": None,
    }


def load_aimet_checkpoint(
    quant_weights: str,
    device: torch.device,
) -> Dict[str, Any]:
    print("[TORCH] No encoding_path provided, trying AIMET checkpoint load...")

    sim = quantsim.load_checkpoint(quant_weights)
    sim.model.to(device).eval()

    return {
        "backend": "torch",
        "model": sim.model,
        "sim": sim,
        "session": None,
        "input_name": None,
        "output_names": None,
        "graph_optimization_level": None,
        "encoding_path": None,
    }


def rebuild_model_with_encodings(
    quant_weights: str,
    encoding_path: str,
    config: Any,
    device: torch.device,
    quant_scheme: str,
    default_output_bw: int,
    default_param_bw: int,
    config_path: Optional[str],
    enable_bn_fold: bool,
    skip_layer_names: Optional[List[str]],
) -> Dict[str, Any]:
    if config is None:
        raise ValueError(
            "When encoding_path is provided, you must also provide config "
            "so the FP32 model can be rebuilt before applying encodings."
        )

    print(f"[TORCH] encoding_path provided: {encoding_path}")
    print("[TORCH] Rebuilding FP32 model and applying encodings...")

    cfg, dataset, data_loader = build_eval_loader(config)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    print(f"[TORCH] Loading exported AIMET weights from: {quant_weights}")
    ckpt = torch.load(quant_weights, map_location="cpu")
    state_dict = ckpt.model.state_dict()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[TORCH] Missing keys: {len(missing)}")
    print(f"[TORCH] Unexpected keys: {len(unexpected)}")
    if missing:
        print("[TORCH] First missing keys:", missing[:20])
    if unexpected:
        print("[TORCH] First unexpected keys:", unexpected[:20])

    model.CLASSES = getattr(dataset, "CLASSES", None)
    model.PALETTE = getattr(dataset, "PALETTE", None)
    model = model.to(device).eval()

    first_batch = prepare_batch(next(iter(data_loader)), device)

    wrapped_model = AimetTraceWrapper(model=model).to(device).eval()
    wrapped_model.set_batch(first_batch)

    real_img = ensure_trace_img_shape(extract_single_img(first_batch))
    dummy_input = torch.zeros_like(real_img)

    maybe_run_bn_fold(wrapped_model, dummy_input, enable_bn_fold)

    sim = create_quant_sim(
        model=wrapped_model,
        device=device,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_path=config_path,
        skip_layer_names=skip_layer_names,
    )

    print("has load_encodings:", hasattr(sim, "load_encodings"))
    sim.load_encodings(encoding_path, strict=True, partial=False)

    log_uninitialized_quantizers(sim)
    sim.model.to(device).eval()

    return {
        "backend": "torch",
        "model": sim.model,
        "sim": sim,
        "session": None,
        "input_name": None,
        "output_names": None,
        "graph_optimization_level": None,
        "encoding_path": encoding_path,
    }


SKIP_LAYER_NAMES = [
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0",
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.dropout",
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.sampling_offsets",
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.attention_weights",
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj",
    # "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0.output_proj",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.dropout",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.sampling_offsets",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.attention_weights",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.value_proj",
    # "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0.output_proj",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.dropout",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.sampling_offsets",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.attention_weights",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.value_proj",
    # "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0.output_proj",
]


def load_quantized_model(
    quant_weights: str,
    device: torch.device,
    provider: str = "CPUExecutionProvider",
    graph_optimization_level: str = "basic",
    encoding_path: Optional[str] = None,
    config: Optional[Any] = None,
    fuse_conv_bn: bool = False,
    quant_scheme: str = "tf_enhanced",
    default_output_bw: int = 8,
    default_param_bw: int = 8,
    config_path: Optional[str] = None,
    enable_bn_fold: bool = False,
) -> Dict[str, Any]:
    
    print("Loading quantized model...")
    ext = os.path.splitext(quant_weights)[1].lower()

    if ext == ".onnx":
        return build_onnx_session(
            quant_weights=quant_weights,
            provider=provider,
            graph_optimization_level=graph_optimization_level,
        )

    if encoding_path is None:
        return load_aimet_checkpoint(
            quant_weights=quant_weights,
            device=device,
        )

    return rebuild_model_with_encodings(
        quant_weights=quant_weights,
        encoding_path=encoding_path,
        config=config,
        device=device,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_path=config_path,
        enable_bn_fold=enable_bn_fold,
        skip_layer_names=SKIP_LAYER_NAMES,
    )