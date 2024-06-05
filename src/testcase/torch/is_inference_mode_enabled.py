
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_inference_mode_enabled)
class TorchIsInferenceModeEnabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_inference_mode_enabled_correctness(self):
        torch.set_grad_enabled(random.choice([True, False]))
        result = torch.is_inference_mode_enabled()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_inference_mode_enabled_large_scale(self):
        torch.set_grad_enabled(random.choice([True, False]))
        result = torch.is_inference_mode_enabled()
        return result

