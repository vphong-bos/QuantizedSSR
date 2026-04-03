from mmcv.cnn.bricks.wrappers import Linear
from aimet_torch.v2.nn import QuantizationMixin
import torch

@QuantizationMixin.implements(Linear)
class QuantizedLinear(QuantizationMixin, Linear):

    def __quant_init__(self):
        super().__quant_init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)