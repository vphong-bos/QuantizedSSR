from script_launcher import run_script

cmd = [
    "/workspace/quant_pipeline/QExporter/model_wrappers/model/ssr/py310_ssr/bin/python",
    "-u",
    "run_eval.py",
    "--config", "/workspace/quant_pipeline/QExporter/model_wrappers/model/ssr/configs/SSR_e2e.py",
    "--fp32_weights", "/workspace/quant_pipeline/QExporter/model_wrappers/model/ssr/data/ckpts/ssr_pt.pth",
    # "--quant_weights", "/kaggle/working/QuantizedSSR/quantized_export/vad_detector_int8/vad_detector_int8_custom_gs.onnx",
    # "--graph_optimization_level", "all",
    "--quant_weights", "/workspace/quant_pipeline/QuantizedSSR/quantized_export/vad_detector_int8/vad_detector_int8.pth",
    "--config_path", "/workspace/quant_pipeline/QuantizedSSR/config/fully_symmetric.json",
    "--encoding_path", "/workspace/quant_pipeline/QuantizedSSR/quantized_export/vad_detector_int8/vad_detector_int8_torch.encodings",
    "--eval", "bbox",
    # "--enable_bn_fold",
]

run_script(
    cmd,
    cwd="/workspace/quant_pipeline/QuantizedSSR",
    label="launcher | eval_model",
    extra_env={
        "WORKING_DIR": "/workspace/quant_pipeline/QuantizedSSR"
    }
)