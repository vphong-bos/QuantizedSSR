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

class AimetTraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.runtime_batch = None

    def set_batch(self, batch):
        self.runtime_batch = batch

    def forward(self, img=None, **kwargs):
        if "return_loss" in kwargs or "rescale" in kwargs or "img_metas" in kwargs:
            return self.model(img=img, **kwargs)

        batch = self.runtime_batch
        assert batch is not None
        return self.model.forward_quant(
            img=img,
            img_metas=batch["img_metas"],
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

@torch.no_grad()
def calibration_forward_pass(model, forward_pass_args):
    dataloader, device, max_batches, max_samples = forward_pass_args
    model.eval()

    seen_samples = 0

    total = len(dataloader)
    if max_batches is not None and max_batches > 0:
        total = min(total, max_batches)

    pbar = tqdm(total=total, desc="Calibration", dynamic_ncols=True)

    for batch_idx, batch in enumerate(dataloader):
        prepared = prepare_batch(batch, device)
        model.set_batch(prepared)

        img = prepared["img"]
        if isinstance(img, list):
            assert len(img) == 1, f"Unexpected img list length: {len(img)}"
            img = img[0]

        _ = model(img)

        current_bs = img.size(0) if torch.is_tensor(img) else 1
        seen_samples += current_bs

        pbar.update(1)
        pbar.set_postfix({
            "samples": seen_samples
        })

        if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
            break
        if max_samples is not None and max_samples > 0 and seen_samples >= max_samples:
            break

    pbar.close()

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

def create_quant_sim(
    model: AimetTraceWrapper,
    device: str,
    dummy_input: torch.Tensor,
    quant_scheme: str,
    default_output_bw: int,
    default_param_bw: int,
    config_path: Optional[str],
    modules_to_ignore,
):
    scheme_map = {
        "tf": QuantScheme.post_training_tf,
        "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    }
    selected_scheme = scheme_map.get(quant_scheme, QuantScheme.post_training_tf_enhanced)

    sim = QuantizationSimModel(
        model=model.to(device).eval(),
        dummy_input=dummy_input,
        quant_scheme=selected_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file=config_path,
        in_place=False,
    )

    sim.exclude_layers_from_quantization(modules_to_ignore)

    return sim

def load_quantized_model(
    quant_weights,
    device,
    provider="CPUExecutionProvider",
    graph_optimization_level="basic",
):
    print("Loading quantized model...")

    ext = os.path.splitext(quant_weights)[1].lower()

    if ext == ".onnx":
        print("Detected ONNX model")

        so = ort.SessionOptions()
        so.graph_optimization_level = get_onnx_graph_optimization_level(
            graph_optimization_level
        )

        session = ort.InferenceSession(
            quant_weights,
            sess_options=so,
            providers=[provider],
        )

        inputs = session.get_inputs()
        outputs = session.get_outputs()

        input_name = inputs[0].name if len(inputs) > 0 else None
        output_names = [o.name for o in outputs]

        print(f"[ONNX] provider: {provider}")
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
            "graph_optimization_level": graph_optimization_level,
        }

    print("Detected AIMET checkpoint")

    sim = quantsim.load_checkpoint(quant_weights)
    sim.model.to(device).eval()

    return {
        "backend": "torch",
        "model": sim.model,
        "session": None,
        "input_name": None,
        "output_names": None,
        "graph_optimization_level": None,
    }