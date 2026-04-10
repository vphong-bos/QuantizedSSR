import time

import torch

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.seq_mse import SeqMseParams, apply_seq_mse

from quantization.quantize_function import AimetTraceWrapper, aimet_forward_fn

# -----------------------------------------------------------------------------
# AIMET stages
# -----------------------------------------------------------------------------
# from mmcv.cnn import fuse_conv_bn
# def maybe_fuse_conv_bn(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
#     if not enabled:
#         print("Conv-BN fusion disabled")
#         return model

#     print("Applying mmcv fuse_conv_bn...")
#     return fuse_conv_bn(model)


def maybe_run_bn_fold(wrapped_model: AimetTraceWrapper, dummy_input: torch.Tensor, enabled: bool) -> None:
    if not enabled:
        print("BN folding disabled")
        return

    print("Applying AIMET batch norm folding...")
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

def maybe_run_cle(wrapped_model: AimetTraceWrapper, dummy_input: torch.Tensor, enabled: bool) -> None:
    if not enabled:
        print("CLE disabled")
        return

    print("Applying Cross-Layer Equalization (CLE)...")
    cle_start = time.time()
    try:
        equalize_model(
            wrapped_model,
            input_shapes=tuple(dummy_input.shape),
            dummy_input=dummy_input,
        )
    except TypeError:
        equalize_model(wrapped_model, dummy_input=dummy_input)
    cle_time = time.time() - cle_start
    print(f"CLE finished in {cle_time:.2f} s")


def maybe_run_seq_mse(
    wrapped_model: AimetTraceWrapper,
    sim: QuantizationSimModel,
    calib_loader,
    enabled: bool,
    num_batches: int,
) -> None:
    if not enabled:
        print("Sequential MSE disabled")
        return

    print("Applying Sequential MSE...")
    params = SeqMseParams(
        num_batches=min(num_batches, len(calib_loader)),
        forward_fn=aimet_forward_fn,
    )

    try:
        apply_seq_mse(
            model=wrapped_model,
            sim=sim,
            data_loader=calib_loader,
            params=params,
            modules_to_exclude=None,
        )
    except TypeError:
        apply_seq_mse(
            model=wrapped_model,
            dummy_input=None,
            data_loader=calib_loader,
            params=params,
            forward_fn=aimet_forward_fn,
        )

    print("Sequential MSE finished.")