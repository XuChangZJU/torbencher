
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.L1Loss)
class TorchNNL1LossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_l1_loss(self):
        
        a = torch.randn(3, 5, requires_grad=True)
        target = torch.empty(3, 5).random_(5)
        loss = torch.nn.L1Loss()
        result = loss(a, target)
        return result

