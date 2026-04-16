from typing import Union, Optional, List, Any, Dict
import warnings
import torch
import copy
import numpy as np
import pandas as pd
import time


def _clone_to_device(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        y = x.detach().clone()
        if dtype is not None:
            y = y.to(dtype=dtype)
        if device is not None:
            y = y.to(device)
        return y
    if isinstance(x, np.ndarray):
        y = torch.from_numpy(x).clone()
        if dtype is not None:
            y = y.to(dtype=dtype)
        if device is not None:
            y = y.to(device)
        return y
    return x


def to_l1(input):
    """Torch-only no-op kept for API compatibility."""
    if isinstance(input, list):
        return [to_l1(x) for x in input]
    elif isinstance(input, dict):
        return {k: to_l1(v) for k, v in input.items()}
    elif isinstance(input, torch.Tensor):
        return input
    else:
        warnings.warn(f"unsupported type: {type(input)}. return the input as is.")
        return input


def to_dram(input):
    """Torch-only no-op kept for API compatibility."""
    if isinstance(input, list):
        return [to_dram(x) for x in input]
    elif isinstance(input, dict):
        return {k: to_dram(v) for k, v in input.items()}
    elif isinstance(input, torch.Tensor):
        return input
    else:
        warnings.warn(f"unsupported type: {type(input)}. return the input as is.")
        return input


def tt2pt(input, dtype=None):
    """Convert input(s) to torch tensor(s). Torch-only replacement."""
    if isinstance(input, list):
        return [tt2pt(x, dtype=dtype) for x in input]
    elif isinstance(input, dict):
        return {k: tt2pt(v, dtype=dtype) for k, v in input.items()}
    elif isinstance(input, torch.Tensor):
        return input.to(dtype=dtype) if dtype is not None else input
    elif isinstance(input, np.ndarray):
        out = torch.from_numpy(input)
        return out.to(dtype=dtype) if dtype is not None else out
    else:
        warnings.warn(f"Unsupported type: {type(input)}. Return the input as is.")
        return input


def pt2tt(
    input,
    dtype=None,
    tile=None,
    pad_value=None,
    layout=None,
    device=None,
    memory_config=None,
    mesh_mapper=None,
):
    """
    Torch-only replacement for pt2tt.
    Keeps signature for compatibility; TT-specific args are ignored.
    """
    if tile is not None:
        warnings.warn("tile is ignored in torch-only mode.")
    if pad_value is not None:
        warnings.warn("pad_value argument is ignored in pt2tt; use pad utilities explicitly.")
    if layout is not None:
        warnings.warn("layout is ignored in torch-only mode.")
    if memory_config is not None:
        warnings.warn("memory_config is ignored in torch-only mode.")
    if mesh_mapper is not None:
        warnings.warn("mesh_mapper is ignored in torch-only mode.")

    kwargs = {
        "dtype": dtype,
        "device": device,
    }

    if isinstance(input, list):
        return [pt2tt(x, **kwargs) for x in input]
    elif isinstance(input, dict):
        return {k: pt2tt(v, **kwargs) for k, v in input.items()}
    elif isinstance(input, (torch.Tensor, np.ndarray)):
        return _clone_to_device(input, device=device, dtype=dtype)
    else:
        warnings.warn(f"Unsupported type: {type(input)}. Return the input as is.")
        return input


def deepcopy_with_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone().detach()
    elif isinstance(obj, list):
        return [deepcopy_with_tensors(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: deepcopy_with_tensors(value) for key, value in obj.items()}
    else:
        return copy.deepcopy(obj)


def setup_host_input(torch_input, img_config, cq_id=0):
    """
    Split the original tensor into a list of sub-tensors corresponding to the images.
    Torch-only version:
      - Takes first batch item.
      - Converts from NCHW to NHWC at the camera/image level.
      - Reshapes each image to (1, 1, H*W, C).
      - Pads channel dim to img_config["padding"] if needed.
    """
    _ = cq_id  # unused in torch mode

    device = img_config.get("device", None)
    dtype = img_config.get("dtype", None)
    pad_channels_to = img_config.get("padding", None)

    # Original code: torch_input = torch_input[0].permute(0, 2, 3, 1)
    # Assumes input shape like [batch, num_cams, C, H, W]
    torch_input = torch_input[0].permute(0, 2, 3, 1)  # [num_cams, H, W, C]

    input_list = []
    for i in range(torch_input.shape[0]):
        inp = torch_input[i : i + 1]  # [1, H, W, C]
        if dtype is not None:
            inp = inp.to(dtype=dtype)
        if device is not None:
            inp = inp.to(device)

        inp = inp.reshape(1, 1, inp.shape[0] * inp.shape[1] * inp.shape[2], inp.shape[3])

        if pad_channels_to is not None and pad_channels_to > inp.shape[-1]:
            pad_c = pad_channels_to - inp.shape[-1]
            inp = torch.nn.functional.pad(inp, (0, pad_c, 0, 0, 0, 0, 0, 0), value=0)

        input_list.append(inp)

    return input_list


def setup_dram_sharded_config(ttnn_host_input, device):
    """Torch-only compatibility stub."""
    _ = ttnn_host_input, device
    return None


def setup_l1_sharded_config(ttnn_host_input, device):
    """Torch-only compatibility stub."""
    _ = ttnn_host_input, device
    return None


def setup_dram_input(ttnn_tensors, img_config, device, cq_id=0):
    """
    Torch-only replacement:
    move/clone tensors to target device.
    """
    _ = cq_id  # unused
    num_cams = img_config["num_cams"]
    input_device = [_clone_to_device(ttnn_tensors[i], device=device) for i in range(num_cams)]
    return input_device


def _extract_container_field(data, key, default=None):
    if key not in data:
        return default
    value = data[key]
    try:
        # Typical MMCV DataContainer usage from your original file
        return value[0].data
    except Exception:
        try:
            return value.data[0]
        except Exception:
            return value


def extract_data_from_container(
    data,
    tensor="pt",
    output_storage=None,
    device=None,
    input_config=None,
    cq_id=0,
    **kwargs,
):
    """
    Torch-only extraction from MMCV-style DataContainer.
    'tt' is treated as a compatibility alias for torch mode.
    """
    _ = output_storage, cq_id, kwargs  # unused in torch mode

    if tensor in ("pt", "torch"):
        data["img_metas"] = _extract_container_field(data, "img_metas", None)
        data["gt_bboxes_3d"] = _extract_container_field(data, "gt_bboxes_3d", None)
        data["gt_labels_3d"] = _extract_container_field(data, "gt_labels_3d", None)
        data["img"] = _extract_container_field(data, "img", None)
        data["ego_his_trajs"] = _extract_container_field(data, "ego_his_trajs", None)
        data["ego_fut_trajs"] = _extract_container_field(data, "ego_fut_trajs", None)
        data["ego_fut_cmd"] = _extract_container_field(data, "ego_fut_cmd", None)
        data["ego_lcf_feat"] = _extract_container_field(data, "ego_lcf_feat", None)
        data["gt_attr_labels"] = _extract_container_field(data, "gt_attr_labels", None)

        if "map_gt_labels_3d" in data:
            try:
                data["map_gt_labels_3d"] = data["map_gt_labels_3d"].data[0]
            except Exception:
                data["map_gt_labels_3d"] = data["map_gt_labels_3d"]

        if "map_gt_bboxes_3d" in data:
            try:
                data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"].data[0]
            except Exception:
                data["map_gt_bboxes_3d"] = data["map_gt_bboxes_3d"]

        # Optional device move for common tensor fields
        if device is not None:
            for key in [
                "img",
                "ego_his_trajs",
                "ego_fut_trajs",
                "ego_fut_cmd",
                "ego_lcf_feat",
                "gt_attr_labels",
            ]:
                if isinstance(data.get(key), torch.Tensor):
                    data[key] = data[key].to(device)

            # img_metas often contains nested numpy arrays / tensors
            if isinstance(data.get("img_metas"), list):
                pass

        return data
    else:
        raise RuntimeError(f"Undefined tensor type for dataloader extraction: {tensor}")


def preprocess_dataloader(dataloader):
    """Preprocess all samples in the dataloader."""
    preprocessed_data = []
    for data in dataloader:
        preprocessed_data.append(extract_data_from_container(data, tensor="torch"))
    return preprocessed_data


def compare_tensors(
    x: Union[str, torch.Tensor],
    y: Union[str, torch.Tensor],
    message: Optional[str] = None,
    permute: Optional[List[int]] = None,
    debug: Optional[bool] = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    if not debug:
        return True

    if message is not None:
        print(message, end=" ")

    def load_tensor(obj):
        if isinstance(obj, str):
            if obj.endswith(".pt"):
                return torch.load(obj, map_location="cpu")
            elif obj.endswith(".npy"):
                return torch.from_numpy(np.load(obj))
            else:
                raise ValueError(f"Unknown file type for: {obj}")
        return obj

    try:
        x = load_tensor(x)
        y = load_tensor(y)
    except Exception:
        print("Loading tensors error, skip comparison.")
        return False

    x = tt2pt(x)
    y = tt2pt(y)

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Inputs must be or convert to torch.Tensor")

    if permute:
        y = y.permute(*permute)

    if x.numel() != y.numel():
        print(f"Numel mismatch: {x.numel()} vs {y.numel()}")
        return False

    x = x.to(dtype=y.dtype).reshape(y.shape)

    same = torch.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not same:
        diff = (x - y).abs()
        print(
            f"FAILED | max_abs_diff={diff.max().item():.6g}, "
            f"mean_abs_diff={diff.mean().item():.6g}"
        )
    else:
        print("OK")
    return same


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    axis: int,
    pad_value=0,
):
    """
    Pad a torch tensor along one axis to the next multiple.
    Returns (padded_tensor, pad_amount).
    """
    length = tensor.shape[axis]
    pad = (multiple - length % multiple) % multiple
    if pad == 0:
        return tensor, 0

    pad_shape = list(tensor.shape)
    pad_shape[axis] = pad
    extra = torch.full(
        pad_shape,
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    tensor = torch.cat([tensor, extra], dim=axis)
    return tensor, pad


def masked_fill(tensor, mask, value):
    """Fills elements of the input tensor with value where mask is True."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor)}")

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    elif not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask)

    mask = mask.to(device=tensor.device)
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    return tensor.masked_fill(mask, value)