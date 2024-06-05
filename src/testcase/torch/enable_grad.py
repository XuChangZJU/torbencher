
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.enable_grad)
class TorchEnableGradTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_enable_grad_correctness(self):
        x = torch.randn(random.randint(1, 10), requires_grad=False)
        with torch.enable_grad():
            y = x * 2
        result = y.requires_grad
        return result

    @test_api_version.larger_than("1.1.3")
    def test_enable_grad_large_scale(self):
        x = torch.randn(random.randint(1000, 10000), requires_grad=False)
        with torch.enable_grad():
            y = x * 2
        result = y.requires_grad
        return result

@test_api(torch.autograd.grad_mode.set_grad_enabled)
class TorchSetGradEnabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_grad_enabled_correctness(self):
        x = torch.randn(random.randint(1, 10), requires_grad=True)
        is_train = random.choice([True, False])
        with torch.autograd.grad_mode.set_grad_enabled(is_train):
            y = x * 2
        result = y.requires_grad
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_grad_enabled_large_scale(self):
        x = torch.randn(random.randint(1000, 10000), requires_grad=True)
        is_train = random.choice([True, False])
        with torch.autograd.grad_mode.set_grad_enabled(is_train):
            y = x * 2
        result = y.requires_grad
        return result

