import torch

import torch.nn as nn

from evaluation.eval_dataset import extract_data

from aimet_torch import quantsim
from aimet_common.defs import QuantScheme
from aimet_torch.quantsim import QuantizationSimModel

from quantization.registered_ops import QuantizedLinear

from typing import Optional, Dict, Any

import pickle
import torch
import numpy as np
import torch.nn as nn

import torch
import torch.nn as nn

class AimetTraceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.runtime_batch = None

    def set_batch(self, batch):
        self.runtime_batch = batch

    def forward(self, *args, return_loss=False, rescale=True, **kwargs):
        """
        Supports both:
          1) AIMET tracing:
               wrapper(images)
          2) Normal detector eval:
               wrapper(return_loss=False, rescale=True, **data)
        """

        # Case 1: AIMET trace / QuantSim dummy_input path
        if len(args) == 1 and torch.is_tensor(args[0]):
            if self.runtime_batch is None:
                raise RuntimeError(
                    "AimetTraceWrapper.forward(images) called before set_batch()."
                )

            batch = dict(self.runtime_batch)
            batch["img"] = args[0]
            outputs = self.model(return_loss=False, rescale=True, **batch)
            return self._make_traceable_output(outputs)

        # Case 2: Normal eval path with full detector batch
        if kwargs:
            outputs = self.model(return_loss=return_loss, rescale=rescale, **kwargs)
            return outputs

        # Case 3: fallback to cached batch
        if self.runtime_batch is not None:
            outputs = self.model(
                return_loss=False,
                rescale=True,
                **self.runtime_batch,
            )
            return self._make_traceable_output(outputs)

        raise RuntimeError("No valid inputs were provided to AimetTraceWrapper.forward().")

    def _make_traceable_output(self, outputs):
        if torch.is_tensor(outputs):
            return outputs

        if isinstance(outputs, dict):
            tensor_dict = {k: v for k, v in outputs.items() if torch.is_tensor(v)}
            if len(tensor_dict) == 1:
                return next(iter(tensor_dict.values()))
            if tensor_dict:
                return tuple(tensor_dict.values())

        if isinstance(outputs, (list, tuple)):
            gathered = []
            for item in outputs:
                if torch.is_tensor(item):
                    gathered.append(item)
                elif isinstance(item, dict):
                    for value in item.values():
                        if torch.is_tensor(value):
                            gathered.append(value)

            if len(gathered) == 1:
                return gathered[0]
            if gathered:
                return tuple(gathered)

        raise RuntimeError(
            "Model output does not expose tensors suitable for AIMET tracing."
        )
    
def aimet_forward_fn(model, data):
    return model(return_loss=False, rescale=True, **data)
# import copy

# def aimet_forward_fn(model, img, static_inputs):
#     if not isinstance(static_inputs, dict):
#         raise TypeError(f"static_inputs must be dict, got {type(static_inputs)}")

#     data = copy.copy(static_inputs)
#     data["img"] = data

#     return model(return_loss=False, rescale=True, **data)

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
def run_model_on_batch(model: AimetTraceWrapper, batch: Dict[str, Any], device: torch.device):
    batch = prepare_batch(batch, device)
    model.set_batch(batch)
    return model(batch["img"])

@torch.no_grad()
def calibration_forward_pass(model, forward_pass_args):
    dataloader, device, max_batches, max_samples = forward_pass_args
    model.eval()

    seen_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            prepared = prepare_batch(batch, device)

            if hasattr(model, "set_batch"):
                model.set_batch(prepared)
                _ = model(prepared["img"])
            else:
                _ = model(prepared["img"])

            batch_img = prepared["img"]
            if isinstance(batch_img, list):
                batch_img = batch_img[0]

            current_bs = batch_img.size(0) if torch.is_tensor(batch_img) else 1
            seen_samples += current_bs

            if max_batches is not None and max_batches > 0 and batch_idx + 1 >= max_batches:
                break
            if max_samples is not None and max_samples > 0 and seen_samples >= max_samples:
                break

import os
import torch
import onnxruntime as ort

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
    return sim

def load_quantized_model(
    quant_weights,
    device,
    provider="CPUExecutionProvider",
    graph_optimization_level="basic",
):
    print("Loading quantized model...")

    ext = os.path.splitext(quant_weights)[1].lower()

    # =========================
    # Case 1: ONNX QDQ model
    # =========================
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

        input_name = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]

        print(f"[ONNX] provider: {provider}")
        print(f"[ONNX] graph optimization level: {graph_optimization_level}")
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
        "graph_optimization_level": None,
    }
