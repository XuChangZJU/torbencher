
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.smooth_l1_loss)
class SmoothL1LossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss_correctness(self):
        input_data = torch.randn(10, 10)
        target = torch.randn(10, 10)
        beta = random.uniform(0.0, 1.0)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.smooth_l1_loss(input_data, target, beta, reduction)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_smooth_l1_loss_large_scale(self):
        input_data = torch.randn(1000, 1000)
        target = torch.randn(1000, 1000)
        beta = random.uniform(0.0, 1.0)
        reduction = random.choice(['mean', 'sum', 'none'])
        result = torch.nn.functional.smooth_l1_loss(input_data, target, beta, reduction)
        return result

