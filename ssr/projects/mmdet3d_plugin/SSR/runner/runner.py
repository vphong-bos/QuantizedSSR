from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from ssr.projects.mmdet3d_plugin.SSR.runner.runner_infra import (
    DoubleStorage,
    InvocationBuffer,
    SSRRunnerInfra,
)
from ssr.visualization.visualization import Visualizer
from ssr.visualization.bevmaploader import BEVMapLoader


class SSRRunner:
    def __init__(
        self,
        device: Any,
        fp32_checkpoint_path: Optional[str] = None,
        quant_checkpoint_path: Optional[str] = None,
        encodings_path: Optional[str] = None,
        config_path: Optional[Any] = None,
        config: Optional[str] = None,
        double_cq: bool = False,
        dataset: Optional[Any] = None,
        enable_bn_fold: Optional[Any] = None,
    ) -> None:
        # Basic parameters
        self.device = device
        self.config_path = config_path
        self.config = config

        # Double CQ
        self.double_cq = double_cq
        self.start_compute_event = []
        self.end_compute_event = []
        self.finish_write_event = []
        self.finish_read_event = []
        self.io_cq_id = 1 if double_cq else 0
        self.compute_cq_id = 0

        # Runner infrastructure
        self.runner_infra = SSRRunnerInfra(
            device=self.device,
            config_path=self.config_path,
            config=self.config,
            fp32_checkpoint_path=fp32_checkpoint_path,
            quant_checkpoint_path=quant_checkpoint_path,
            encodings_path=encodings_path,
            enable_bn_fold=enable_bn_fold,
        )

        # I/O storage
        self.input_storage = DoubleStorage() if self.double_cq else None
        self.output_storage: Dict[str, Any] = {}
        self.host_output: Dict[str, Any] = {}

        # Scheduler
        self.sched = InvocationBuffer(device=self.device, target_cq_id=self.io_cq_id)

        # Post-process helpers
        self.dataset = dataset
        self.visualizer: Optional[Visualizer] = None
        self.bevloader: Optional[BEVMapLoader] = None

        # Timing
        self.execution_time = 0.0
        self.last_time = time.time()

    # ----------------------------- Small utilities ---------------------------

    def _record_execution_time(self) -> None:
        """Update self.execution_time from the last timestamp and move the cursor."""
        end_time = time.time()
        self.execution_time = end_time - getattr(self, "last_time", 0.0)
        self.last_time = end_time

    def _should_post_process(self, **kwargs: Any) -> bool:
        return kwargs.get("visualize", False) or kwargs.get("validate", False)

    # ----------------------------- Core modes --------------------------------

    def _reference_run(self, data: Dict[str, Any]) -> Any:
        """Run PyTorch reference (no gradients)."""
        data_pt = extract_data_from_container(deepcopy(data), tensor="pt")
        with torch.no_grad():
            self.runner_infra.run_ref(data_pt)
        return self.runner_infra.ref_out

    # ----------------------------- Paths & Post -------------------------------

    def get_output_path(self, out_dir: str = "generated/video") -> str:
        """Construct a dated mp4 path under `out_dir` (absolute if needed)."""
        current_path = os.getcwd()
        current_path = current_path.rsplit("/", 1)[0] + "/"
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(current_path, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(out_dir, f"realtime_{ts}.mp4")

    def _post_process(self, tt_result: Dict[str, Any], data: Dict[str, Any], **kwargs: Any) -> None:
        """Visualization/validation path (batch size currently restricted to 1)."""
        bbox_results = self.simple_test(
            outs=tt_result,
            img_metas=(data["img_metas"][0].data)[0],
            ego_fut_cmd=(data["ego_fut_cmd"][0].data)[0],
        )
        tt_result = bbox_results

        # if kwargs.get("visualize", False) and kwargs.get("use_bev", False):
        #     st_bev = time.time()
        #     sample = self.bevloader.get_batch_bev_map_info(kwargs.get("sample_idx", 0))
        #     self.bevloader.draw_batch_bev_map(sample)
        #     logger.debug(f"Done draw BEV Map, taken: {time.time() - st_bev:.4f}")

        if kwargs.get("visualize", False):
            v_st = time.time()
            visualize_result = self.dataset._format_bbox_realtime(
                {"bbox_results": tt_result}["bbox_results"], is_ttnn=True, sample_idx=kwargs.get("sample_idx", 0)
            )
            self.visualizer.create_visual(
                index=kwargs.get("sample_idx", 0),
                visualize_result=visualize_result,
                car_path=os.path.join(os.getenv("TT_METAL_HOME"), "models/bos_model/ssr/scripts/assets/car2.png"),
                logo_path=os.path.join(os.getenv("TT_METAL_HOME"), "models/bos_model/ssr/scripts/assets/bos_logo.png"),
                fps=round(1.0 / self.execution_time, 2),
            )
            if kwargs.get("use_bev", False):
                visualize_img = self.visualizer.get_visual()
                self.bevloader.get_token_id(visualize_result)
                new_visual = self.bevloader.create_output_bev_map(canvas=visualize_img)
                self.visualizer.set_visual(new_visual)

            if kwargs.get("realtime", False):
                keep_running = self.visualizer.show_realtime()
                if not keep_running:
                    logger.info("Realtime display stopped by user.")
            else:
                self.visualizer.show_video()

    # ----------------------------- Public API --------------------------------

    def dealloc_output(self) -> None:
        self.runner_infra.dealloc_output()

    def close_visualizer(self, **kwargs: Any) -> None:
        if kwargs.get("realtime", False):
            self.visualizer.close_realtime()
            logger.info("Close window")
        else:
            self.visualizer.close_video()
            logger.info(f"Video saved at {self.visualizer.video_path}")

    def simple_test(
        self,
        outs: Dict[str, Any],
        img_metas: Any,
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Test function without augmentation."""
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts, metric_dict = self.simple_test_pts(outs, ego_fut_cmd=ego_fut_cmd)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict
        return bbox_list
    
    def simple_test_pts(
        self,
        outs: Dict[str, Any],
        ego_fut_cmd: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """Test function (batch_size=1 only)."""
        bbox_results = []
        for i in range(len(outs["ego_fut_preds"])):
            bbox_result = dict()
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i]
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "only support batch_size=1 now"
        metric_dict = None
        return bbox_results, metric_dict

    def run(self, data: Dict[str, Any], mode: str = "performant", **kwargs: Any) -> Any:
        # Lazy-init visualization utilities if needed
        if kwargs.get("visualize", False):
            if self.visualizer is None:
                self.visualizer = Visualizer(dataset=self.dataset)
                if not kwargs.get("realtime", False):
                    self.visualizer.set_output(self.get_output_path())
            if kwargs.get("use_bev", False) and self.bevloader is None:
                self.bevloader = BEVMapLoader(dataset=self.dataset)
                logger.info("Pre-loading BEV map from dataset")
                self.bevloader.draw_bev_map()

        output = None
        if mode == "reference":
            output = self._reference_run(data)

        return output

    def __call__(self, *args, **kwds):
        return self.run(*args, **kwds)