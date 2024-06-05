
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_grad_enabled)
class TorchIsGradEnabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_grad_enabled_correctness(self):
        torch.set_grad_enabled(random.choice([True, False]))
        result = torch.is_grad_enabled()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_grad_enabled_large_scale(self):
        torch.set_grad_enabled(random.choice([True, False]))
        result = torch.is_grad_enabled()
        return result

@test_api(torch.autograd.grad_mode.inference_mode)
class TorchInferenceModeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inference_mode_correctness(self):
        x = torch.randn(random.randint(1, 10), requires_grad=True)
        with torch.autograd.grad_mode.inference_mode():
            y = x * 2
        result = y.requires_grad
        return result

    @test_api_version.larger_than("1.1.3")
    def test_inference_mode_large_scale(self):
        x = torch.randn(random.randint(1000, 10000), requires_grad=True)
        with torch.autograd.grad_mode.inference_mode():
            y = x * 2
        result = y.requires_grad
        return result

