
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.KLDivLoss)
class TorchKLDivLossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kldivloss_correctness(self):
        input_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10)).log_softmax(dim=1)
        target = torch.randn(random.randint(1, 10), random.randint(1, 10)).log_softmax(dim=1)
        kl_div_loss = torch.nn.KLDivLoss()
        result = kl_div_loss(input_tensor, target)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_kldivloss_large_scale(self):
        input_tensor = torch.randn(random.randint(1000, 10000), random.randint(100, 1000)).log_softmax(dim=1)
        target = torch.randn(random.randint(1000, 10000), random.randint(100, 1000)).log_softmax(dim=1)
        kl_div_loss = torch.nn.KLDivLoss()
        result = kl_div_loss(input_tensor, target)
        return result

