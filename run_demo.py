import os
import sys
import argparse

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_dir)

from script_launcher import run_script


def run_reference_model(args, label="FP32"):
    reference_run = args.reference_run
    if reference_run is None:
        reference_run = os.path.join(args.tt_metal_root, "ssr/bos_model/ssr/reference/run.py")

    cmd = [
        args.python_bin,
        "-u",
        reference_run,
        "--config",
        args.config,
        "--checkpoint",
        args.fp32_weights,
        "--eval",
        args.eval,
    ]

    if args.fuse_conv_bn:
        cmd.append("--fuse-conv-bn")

    run_script(
        cmd,
        cwd=os.path.dirname(reference_run),
        extra_env={"WORKING_DIR": args.working_dir},
        label=label,
    )


def run_quantized_model(args, label="Quantized"):
    quant_eval_script = os.path.join(args.working_dir, "run_eval.py")
    if not os.path.exists(quant_eval_script):
        raise FileNotFoundError(f"Quantized eval script not found: {quant_eval_script}")

    cmd = [
        args.python_bin,
        "-u",
        quant_eval_script,
        "--config",
        args.config,
        "--fp32_weights",
        args.fp32_weights,
        "--quant_weights",
        args.quant_weights,
        "--eval",
        args.eval,
    ]

    if args.config_path:
        cmd.extend(["--config_path", args.config_path])

    if args.encoding_path:
        cmd.extend(["--encoding_path", args.encoding_path])

    if args.enable_bn_fold:
        cmd.append("--enable_bn_fold")

    run_script(
        cmd,
        cwd=args.working_dir,
        extra_env={"WORKING_DIR": args.working_dir},
        label=label,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run demo evaluation for FP32 and quantized SSR models."
    )

    parser.add_argument(
        "task",
        choices=["functional", "performance", "cpu", "demo", "all"],
        help="Task label to run; this script always evaluates both FP32 and quantized models."
    )
    parser.add_argument(
        "--working_dir",
        default="/kaggle/working/QuantizedSSR",
        help="QuantizedSSR project working directory"
    )
    parser.add_argument(
        "--tt_metal_root",
        default="/workspace/tt-metal",
        help="Root directory of the tt-metal checkout containing the reference SSR run script"
    )
    parser.add_argument(
        "--reference_run",
        default=None,
        help="Optional explicit path to the reference TT-Metal SSR run.py script"
    )
    parser.add_argument(
        "--python_bin",
        default="/kaggle/working/py310_ssr/bin/python",
        help="Python executable"
    )
    parser.add_argument(
        "--config",
        default="/kaggle/working/QuantizedSSR/ssr/projects/configs/SSR_e2e.py",
        help="Model config path"
    )
    parser.add_argument(
        "--fp32_weights",
        default="/kaggle/working/QuantizedSSR/ssr/data/ckpts/ssr_pt.pth",
        help="FP32/PT checkpoint path"
    )
    parser.add_argument(
        "--quant_weights",
        default="/kaggle/working/QuantizedSSR/quantized_export/vad_detector_int8/vad_detector_int8.pth",
        help="Quantized PT checkpoint path"
    )
    parser.add_argument(
        "--config_path",
        default="/kaggle/working/QuantizedSSR/config/fully_symmetric.json",
        help="Quantization config json"
    )
    parser.add_argument(
        "--encoding_path",
        default="/kaggle/working/QuantizedSSR/quantized_export/vad_detector_int8/vad_detector_int8_torch.encodings",
        help="Encoding path for quantized PT"
    )
    parser.add_argument(
        "--eval",
        default="bbox",
        help="Eval type"
    )
    parser.add_argument(
        "--fuse_conv_bn",
        action="store_true",
        help="Fuse conv and batchnorm in the reference model"
    )
    parser.add_argument(
        "--enable_bn_fold",
        action="store_true",
        default=True,
        help="Enable BN fold for quantized evaluation"
    )

    args = parser.parse_args()

    label = args.task if args.task != "all" else "all"
    fp32_label = f"{label} | FP32"
    quant_label = f"{label} | Quantized"

    run_reference_model(args, label=fp32_label)
    run_quantized_model(args, label=quant_label)


if __name__ == "__main__":
    main()
