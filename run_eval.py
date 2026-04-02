#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch

# ===== Replace these with your project imports =====
# Example:
# from projects.mmdet3d_plugin.models import build_model
# from tools.some_loader import load_checkpoint_model
# from evaluation.eval_dataset import build_eval_loader
# from evaluation.eval_metrics import evaluate_model
# from utils.pcc_metric import evaluate_pcc
# from utils.export_onnx import export_optimized_onnx_model

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model
from utils.pcc_metric import evaluate_pcc
from utils.export_onnx import export_optimized_onnx_model


DEFAULT_EXPORT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "quantized_export")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", type=str, required=True)

    parser.add_argument("--fp32_weights", type=str, help="Path to FP32 .pth/.pt")
    parser.add_argument("--quant_weights", type=str, help="Path to quantized/exported model: .pth/.pt or .onnx")

    parser.add_argument("--config", type=str, required=True, help="Model config path if needed by your repo")
    parser.add_argument("--checkpoint_key", type=str, default=None, help="Optional key inside checkpoint, e.g. state_dict")

    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--split", type=str, default="val")

    parser.add_argument("--onnx_provider", type=str, default="CPUExecutionProvider",
                        choices=["CPUExecutionProvider", "CUDAExecutionProvider"])

    parser.add_argument("--export_optimized_onnx", action="store_true")
    parser.add_argument("--optimized_onnx_name", type=str, default="optimized_model")

    return parser.parse_args()


def infer_backend(weight_path: str) -> str:
    ext = Path(weight_path).suffix.lower()
    if ext == ".onnx":
        return "onnx"
    if ext in [".pt", ".pth", ".pkl"]:
        return "torch"
    raise ValueError(f"Unsupported model format: {weight_path}")


def build_your_model(config_path: str, device: str):
    """
    Replace this with your repo's model builder.
    Must return a torch.nn.Module.
    """
    # Example:
    # from mmcv import Config
    # from mmdet3d.models import build_model
    # cfg = Config.fromfile(config_path)
    # model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # model.to(device)
    # model.eval()
    # return model

    raise NotImplementedError("Please replace build_your_model() with your repo model builder.")


def load_torch_model(weights_path: str, config_path: str, device: str, checkpoint_key: str = None):
    """
    Generic torch loader for .pth/.pt.
    """
    model = build_your_model(config_path=config_path, device=device)

    checkpoint = torch.load(weights_path, map_location=device)

    if checkpoint_key is not None and checkpoint_key in checkpoint:
        state_dict = checkpoint[checkpoint_key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[Torch] loaded: {weights_path}")
    print(f"[Torch] missing keys: {len(missing)}")
    print(f"[Torch] unexpected keys: {len(unexpected)}")

    model.eval()
    return {
        "backend": "torch",
        "model": model,
        "session": None,
        "input_name": None,
        "output_names": None,
        "model_category_const": None,
    }


def load_onnx_model(weights_path: str, provider: str):
    """
    Generic ONNX loader.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(weights_path, sess_options=sess_options, providers=[provider])

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    print(f"[ONNX] loaded: {weights_path}")
    print(f"[ONNX] provider: {provider}")
    print(f"[ONNX] input : {input_name}")
    print(f"[ONNX] outputs: {output_names}")

    return {
        "backend": "onnx",
        "model": None,
        "session": session,
        "input_name": input_name,
        "output_names": output_names,
        "model_category_const": None,
    }


def load_model_auto(weights_path: str, config_path: str, device: str, provider: str, checkpoint_key: str = None):
    backend = infer_backend(weights_path)
    if backend == "onnx":
        return load_onnx_model(weights_path=weights_path, provider=provider)
    return load_torch_model(
        weights_path=weights_path,
        config_path=config_path,
        device=device,
        checkpoint_key=checkpoint_key,
    )


def main():
    args = parse_args()

    if args.export_optimized_onnx:
        if not args.quant_weights or infer_backend(args.quant_weights) != "onnx":
            raise ValueError("--export_optimized_onnx requires --quant_weights pointing to an .onnx file")

        optimized_onnx_path = os.path.join(DEFAULT_EXPORT_PATH, f"{args.optimized_onnx_name}.onnx")
        export_optimized_onnx_model(
            quant_weights=args.quant_weights,
            output_path=optimized_onnx_path,
            provider=args.onnx_provider,
        )

    loader = build_eval_loader(
        data_root=args.data_root,
        split=args.split,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    fp32_model = None
    fp32_results = None

    if args.fp32_weights:
        print("Loading FP32 model...")
        fp32_obj = load_model_auto(
            weights_path=args.fp32_weights,
            config_path=args.config,
            device=args.device,
            provider=args.onnx_provider,
            checkpoint_key=args.checkpoint_key,
        )

        if fp32_obj["backend"] != "torch":
            raise ValueError("FP32 reference model must be a torch checkpoint (.pth/.pt)")

        fp32_model = fp32_obj["model"]

        print("Evaluating FP32...")
        fp32_results = evaluate_model(
            model_obj=fp32_obj,
            model_category_const=fp32_obj["model_category_const"],
            loader=loader,
            device=args.device,
            max_samples=args.max_samples,
        )
        print(fp32_results)

    if args.quant_weights:
        print("Loading quantized/exported model...")
        quant_obj = load_model_auto(
            weights_path=args.quant_weights,
            config_path=args.config,
            device=args.device,
            provider=args.onnx_provider,
            checkpoint_key=args.checkpoint_key,
        )

        print("Evaluating quantized/exported model...")
        quant_results = evaluate_model(
            model_obj=quant_obj,
            model_category_const=quant_obj["model_category_const"],
            loader=loader,
            device=args.device,
            max_samples=args.max_samples,
        )

        if not args.fp32_weights:
            print(quant_results)

        if args.fp32_weights:
            print("Evaluating PCC between FP32 and quantized/exported outputs...")

            if quant_obj["backend"] == "torch":
                pcc_results = evaluate_pcc(
                    fp32_model=fp32_model,
                    quant_model=quant_obj["model"],
                    loader=loader,
                    device=args.device,
                    max_samples=args.max_samples,
                )
                pcc_value = pcc_results["PCC"]
            else:
                print("Skipping PCC: current evaluate_pcc expects torch-vs-torch, but quant model is ONNX Runtime.")
                pcc_value = float("nan")

            print("\n================ Compare FP32 vs Quantized/Exported ================")
            print("---- Accuracy ----")
            if "mIoU" in fp32_results and "mIoU" in quant_results:
                print(f"FP32  mIoU : {fp32_results['mIoU']:.4f}")
                print(f"INT8  mIoU : {quant_results['mIoU']:.4f}")
                print(f"Drop       : {quant_results['mIoU'] - fp32_results['mIoU']:.4f}")

            print("\n---- Correlation ----")
            print(f"PCC        : {pcc_value:.6f}")

            print("\n---- Performance ----")
            if "FPS" in fp32_results and "FPS" in quant_results:
                print(f"FP32  FPS  : {fp32_results['FPS']:.2f}")
                print(f"INT8  FPS  : {quant_results['FPS']:.2f}")
                print(f"Speedup    : {quant_results['FPS'] / fp32_results['FPS']:.2f}x")

            if "Avg_Inference_Time_ms" in fp32_results and "Avg_Inference_Time_ms" in quant_results:
                print(f"FP32  Latency (ms): {fp32_results['Avg_Inference_Time_ms']:.2f}")
                print(f"INT8  Latency (ms): {quant_results['Avg_Inference_Time_ms']:.2f}")
                print(
                    f"Latency reduction : "
                    f"{fp32_results['Avg_Inference_Time_ms'] - quant_results['Avg_Inference_Time_ms']:.2f} ms"
                )

            print("====================================================================")


if __name__ == "__main__":
    main()