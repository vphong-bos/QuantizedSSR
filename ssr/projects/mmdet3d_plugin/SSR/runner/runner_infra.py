import logging
import os
from ssr.projects.mmdet3d_plugin.SSR.model import load_default_model
from quantization.quantize_function import load_quantized_model

from evaluation.eval_dataset import build_eval_loader

import torch
from mmcv.cnn.bricks.registry import AQuantENTION, NORM_LAYERS
from mmcv.runner import load_checkpoint

from contextlib import contextmanager
import threading

class DoubleStorage:
    def __init__(self):
        self.data = [None, None]
        self.read_idx = 0
        self.write_idx = 0

    def get(self, mode="read"):
        if mode == "read":
            assert self.data[self.read_idx] is not None, "Data not set before get."
            output = self.data[self.read_idx]
            self.read_idx = 1 - self.read_idx
        elif mode == "write":
            output = self.data[self.write_idx]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return output

    def set(self, data):
        self.data[self.write_idx] = data
        return_idx = self.write_idx
        self.write_idx = 1 - self.write_idx
        return return_idx

class InvocationBuffer:
    def __init__(self, device, target_cq_id: int):
        self.device = device
        self.target_cq_id = target_cq_id
        self._blocks = []
        self._tls = threading.local()

    @contextmanager
    def add_block(self, priority=0):
        # fence_id just needs to be monotonic for events recorded on same CQ
        ops = []
        self._tls.ops = ops

        # the function to enqueue work lazily
        def defer(fn):
            # fn must be a zero-arg callable doing the actual work when invoked later
            ops.append(fn)

        try:
            yield defer
        finally:
            self._blocks.append((priority, ops))
            self._tls.ops = None

    def flush(self, num_blocks=99):
        def key(b):
            return b[0]

        self._blocks.sort(key=key)
        for i, (priority, ops) in enumerate(self._blocks):
            # gate the block behind the event on CQ target
            if i >= num_blocks:
                break
            for op in ops:
                op()
        for _ in range(min(num_blocks, len(self._blocks))):
            self._blocks.pop(0)

class SSRRunnerInfra:
    def __init__(
        self,
        device,
        config_path=None,
        config=None,
        fp32_checkpoint_path=None,
        quant_checkpoint_path=None,
        encodings_path=None,
        enable_bn_fold=False,
    ):
        assert config_path is not None, "config is None."
        self.config_path = config_path
        self.cfg, self.dataset, self.data_loader = build_eval_loader(config_path)

        assert quant_checkpoint_path is not None, "quant_checkpoint_path is None."
        if fp32_checkpoint_path is not None:
            assert os.path.exists(fp32_checkpoint_path), f"PT checkpoint path does not exist: {fp32_checkpoint_path}"
        assert os.path.exists(quant_checkpoint_path), f"Quant checkpoint path does not exist: {quant_checkpoint_path}"
        self.config = config
        self.fp32_checkpoint_path = fp32_checkpoint_path
        self.quant_checkpoint_path = quant_checkpoint_path
        self.encodings_path = encodings_path
        self.device = device
        self.num_devices = device.get_num_devices()
        self.enable_bn_fold = enable_bn_fold

        # 2 - Initialize logger
        logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        logging.getLogger("shapely.geos").setLevel(logging.WARNING)

        # 3 - Initialize output tensors
        self.fp32_out = None
        self.quant_out = None

        # 4 - Initialize models
        ## 4.1 - FP32 model
        self.fp32_model, _ = load_default_model(
            cfg=self.cfg,
            checkpoint_path=self.fp32_checkpoint_path,
            dataset=self.dataset,
            fuse_conv_bn=False,
            map_location=self.device,
        )
        self.fp32_model.eval()

        ## 4.2 - Quantized model
        self.quant_obj = load_quantized_model(
            quant_weights=self.quant_checkpoint_path,
            device=self.device,
            encoding_path=self.encodings_path,
            config=self.config,
            config_path=self.config_path,
            enable_bn_fold=self.enable_bn_fold,
        )

    def run_quant(self, data):
        pass

    def run_fp32(self, data):
        if self.fp32_model is None:
            raise ValueError("fp32erence model is not initialized.")
        self.fp32_out = self.fp32_model(**data)

    def validate(self):
        # Compare ego future predictions between Torch and Quant
        assert self.quant_out is not None, "Quant output is None."
        for key in self.quant_out.keys():
            if key == "scene_query":
                continue
            _, pcc = compare_tensors(self.fp32_out[key], self.quant_out[key], 0.98)
            assert pcc > 0.98, f"PCC below threshold: {pcc:.4f}; at key: {key}"
            self.logger.info("SSR test passed (PCC %.4f)", pcc)