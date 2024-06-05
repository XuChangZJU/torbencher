
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.set_grad_enabled)
class TorchAutogradSetGradEnabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_grad_enabled_correctness(self):
        mode = random.choice([True, False])
        result = torch.autograd.set_grad_enabled(mode)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_grad_enabled_large_scale(self):
        mode = random.choice([True, False])
        result = torch.autograd.set_grad_enabled(mode)
        return result


