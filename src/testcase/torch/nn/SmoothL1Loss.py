
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SmoothL1Loss)
class TorchNNSmoothL1LossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss(self, input=None):
        if input is not None:
            result = torch.nn.SmoothL1Loss()(input[0], input[1])
            return result
        a = torch.randn(10, 5, requires_grad=True)
        target = torch.empty(10, 5).random_(5)
        loss = torch.nn.SmoothL1Loss()
        result = loss(a, target)
        return result

