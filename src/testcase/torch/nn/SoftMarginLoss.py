
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SoftMarginLoss)
class TorchSoftMarginLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmarginloss_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        target = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.long)
        soft_margin_loss = torch.nn.SoftMarginLoss()
        result = soft_margin_loss(input_tensor, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_softmarginloss_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        target = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(100, 1000)), dtype=torch.long)
        soft_margin_loss = torch.nn.SoftMarginLoss()
        result = soft_margin_loss(input_tensor, target)
        return result

