import os
import torch
from tqdm import tqdm
from typing import Optional, Dict, Any

import onnxruntime as ort
from aimet_torch import quantsim
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

from quantization.registered_ops import QuantizedLinear
from evaluation.eval_dataset import extract_data

# class AimetTraceWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.runtime_batch = None

#     def set_batch(self, batch):
#         self.runtime_batch = batch

#     def forward(self, img=None, **kwargs):
#         if "return_loss" in kwargs or "rescale" in kwargs or "img_metas" in kwargs:
#             return self.model(img=img, **kwargs)

#         batch = self.runtime_batch
#         assert batch is not None
#         return self.model.forward_quant(
#             img=img,
#             img_metas=batch["img_metas"],
#         )

def unwrap_datacontainer(x):
    while hasattr(x, "data"):
        x = x.data
    return x

class AimetTraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.runtime_batch = None

    def set_batch(self, batch):
        self.runtime_batch = batch

    def forward(self, img=None, **kwargs):
        # Real path: used by eval / compute_encodings
        if kwargs:
            return self.model(img=img, **kwargs)

        # Trace path: used by QuantizationSimModel(dummy_input=...)
        batch = self.runtime_batch
        assert batch is not None, "runtime_batch must be set before tracing"

        if img is None:
            raise ValueError("img must not be None in trace mode")

        if img.ndim == 4:
            img = img.unsqueeze(0)
        elif img.ndim != 5:
            raise ValueError(f"Unexpected img shape in trace mode: {img.shape}")

        return self.model.forward_trace(
            img_metas=batch["img_metas"],
            img=[img],  # preserve your extracted batch schema
            ego_fut_cmd=batch["ego_fut_cmd"],
            ego_his_trajs=batch["ego_his_trajs"],
            ego_lcf_feat=batch["ego_lcf_feat"],
        )

def aimet_forward_fn(model, inputs):
    if isinstance(inputs, dict):
        img = inputs["img"]
    elif isinstance(inputs, (list, tuple)):
        img = inputs[0]
    else:
        img = inputs

    if isinstance(img, list):
        assert len(img) == 1, f"Unexpected img list length: {len(img)}"
        img = img[0]

    img = img.to(next(model.parameters()).device)
    return model(img)

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
    batch = extract_data(batch)
    batch = move_to_device(batch, device)
    return batch

@torch.no_grad()
def run_model_on_batch(model, batch, device):
    batch = prepare_batch(batch, device)
    model.set_batch(batch)

    img = batch["img"]
    if isinstance(img, list):
        assert len(img) == 1, f"Unexpected img list length: {len(img)}"
        img = img[0]

    return model(img)

def move_to_device_keep_structure(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [move_to_device_keep_structure(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device_keep_structure(v, device) for v in x)
    if isinstance(x, dict):
        return {k: move_to_device_keep_structure(v, device) for k, v in x.items()}
    return x

def calibration_forward_pass(model, forward_pass_args):
    data_loader, device, calib_batches, calib_max_samples = forward_pass_args

    model.eval()
    seen = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = extract_data(data)  # keep unchanged
            data = move_to_device_keep_structure(data, device)

            model(return_loss=False, rescale=True, **data)

            batch_img = data["img"][0]
            batch_size = batch_img.shape[0]
            seen += batch_size

            if calib_batches is not None and i + 1 >= calib_batches:
                break
            if calib_max_samples is not None and seen >= calib_max_samples:
                break

# @torch.no_grad()
# def calibration_forward_pass(model, forward_pass_args):
#     dataloader, device, max_batches, max_samples = forward_pass_args
#     model.eval()

#     seen_samples = 0

#     total = len(dataloader)
#     if max_batches is not None and max_batches > 0:
#         total = min(total, max_batches)

#     pbar = tqdm(total=total, desc="Calibration", dynamic_ncols=True)

#     for batch_idx, batch in enumerate(dataloader):
#         prepared = prepare_batch(batch, device)
#         model.set_batch(prepared)

#         img = prepared["img"]
#         if isinstance(img, list):
#             assert len(img) == 1, f"Unexpected img list length: {len(img)}"
#             img = img[0]

#         _ = model(img)

#         current_bs = img.size(0) if torch.is_tensor(img) else 1
#         seen_samples += current_bs

#         pbar.update(1)
#         pbar.set_postfix({
#             "samples": seen_samples
#         })

#         if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
#             break
#         if max_samples is not None and max_samples > 0 and seen_samples >= max_samples:
#             break

#     pbar.close()

def get_onnx_graph_optimization_level(level):
    """
    Map a user-friendly string or enum-like value to ONNX Runtime optimization level.
    Supported values:
        - "disable_all"
        - "basic"
        - "extended"
        - "all"
    """
    if isinstance(level, str):
        key = level.strip().lower()
        mapping = {
            "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        if key not in mapping:
            raise ValueError(
                f"Unsupported graph_optimization_level: {level}. "
                f"Choose from {list(mapping.keys())}"
            )
        return mapping[key]

    return level

from typing import Optional, Iterable, List, Tuple
import torch
import torch.nn as nn

from aimet_common.defs import QuantScheme
from aimet_torch.v2.quantsim import QuantizationSimModel
from aimet_torch.v2.nn import QuantizationMixin


def get_named_modules_to_ignore(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    exclude_keywords = [
        "positional_encoding",
        "embedding",
        "norm",
        "reference_points",
        "map_reference_points",
        "attention_weights",
    ]

    ignored = []
    for name, module in model.named_modules():
        if any(k in name for k in exclude_keywords):
            ignored.append((name, module))
    return ignored

def apply_quantmixin_ignore(modules_to_ignore):
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
    model,
    device,
    dummy_input,
    quant_scheme,
    default_output_bw,
    default_param_bw,
    config_path=None,
    skip_layer_names=None,
):
    scheme_map = {
        "tf": QuantScheme.post_training_tf,
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    }
    selected_scheme = scheme_map.get(
        quant_scheme, QuantScheme.post_training_tf_enhanced
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

    if skip_layer_names is not None:
        for name in skip_layer_names:
            if name in name_to_module:
                layers_to_exclude.append(name_to_module[name])
                print(f"[EXCLUDE LAYER] {name}")
            else:
                missing.append(name)

    if missing:
        print("Missing layer names:")
        for name in missing:
            print("  ", name)

    if layers_to_exclude:
        sim.exclude_layers_from_quantization(layers_to_exclude)

    return sim

def load_quantized_model(
    quant_weights,
    device,
    provider="CPUExecutionProvider",
    graph_optimization_level="basic",
    encoding_path=None,
    config=None,
    fuse_conv_bn=False,
    quant_scheme="tf_enhanced",
    default_output_bw=8,
    default_param_bw=8,
    config_path=None,
):
    print("Loading quantized model...")

    ext = os.path.splitext(quant_weights)[1].lower()

    if ext == ".onnx":
        print("Detected ONNX model")

        so = ort.SessionOptions()
        so.graph_optimization_level = get_onnx_graph_optimization_level(
            graph_optimization_level
        )

        available = ort.get_available_providers()
        print(f"[ONNX] available providers: {available}")

        if provider == "TensorrtExecutionProvider":
            providers = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        elif provider == "CUDAExecutionProvider":
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]

        providers = [p for p in providers if p in available]

        print(f"[ONNX] requested provider: {provider}")
        print(f"[ONNX] provider fallback order: {providers}")

        session = ort.InferenceSession(
            quant_weights,
            sess_options=so,
            providers=providers,
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        input_name = inputs[0].name if len(inputs) > 0 else None
        output_names = [o.name for o in outputs]

        print(f"[ONNX] session providers: {session.get_providers()}")
        print(f"[ONNX] graph optimization level: {graph_optimization_level}")
        print(f"[ONNX] num_inputs: {len(inputs)}")
        print(f"[ONNX] input: {input_name}")
        print(f"[ONNX] outputs: {output_names}")

        return {
            "backend": "onnx",
            "session": session,
            "input_name": input_name,
            "output_names": output_names,
            "model": None,
            "sim": None,
            "graph_optimization_level": graph_optimization_level,
            "encoding_path": encoding_path,
        }

    # Torch / AIMET path
    print("Detected torch/AIMET artifact")

    # Case 1: real AIMET checkpoint created by quantsim.save_checkpoint(...)
    if encoding_path is None:
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

    # Case 2: plain model checkpoint + encodings file
    print(f"[TORCH] encoding_path provided: {encoding_path}")
    print("[TORCH] Rebuilding FP32 model and applying encodings...")

    if config is None:
        raise ValueError(
            "When encoding_path is provided, you must also provide config and checkpoint "
            "so the FP32 model can be rebuilt before applying encodings."
        )

    from evaluation.eval_dataset import build_eval_loader
    from ssr.projects.mmdet3d_plugin.SSR.model import load_default_model

    from ssr.projects.mmdet3d_plugin.SSR.utils.builder import build_model

    cfg, dataset, data_loader = build_eval_loader(config)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))

    print(f"[TORCH] Loading exported AIMET weights from: {quant_weights}")
    ckpt = torch.load(quant_weights, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

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

    first_batch = next(iter(data_loader))
    first_batch = extract_data(first_batch)
    prepared_batch = move_to_device_keep_structure(first_batch, torch.device(device))

    wrapped_model = AimetTraceWrapper(model=model).to(device).eval()
    wrapped_model.set_batch(prepared_batch)

    real_img = prepared_batch["img"][0]
    if not torch.is_tensor(real_img):
        raise TypeError(f"Expected tensor, got {type(real_img)}")

    if real_img.ndim == 4:
        real_img = real_img.unsqueeze(0)
    elif real_img.ndim != 5:
        raise ValueError(f"Unexpected real_img shape: {real_img.shape}")

    dummy_input = torch.zeros_like(real_img)

    # Keep this aligned with your PTQ build script skip list
    skip_layer_names = [
        "model.pts_bbox_head.transformer.encoder.layers.0.attentions.0",
        "model.pts_bbox_head.transformer.encoder.layers.1.attentions.0",
        "model.pts_bbox_head.transformer.encoder.layers.2.attentions.0",
    ]

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

    print(f"[TORCH] Loading encodings from: {encoding_path}")
    sim.set_and_freeze_param_encodings(encoding_path)

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