
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.inference_mode)
class TorchAutogradInferenceModeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_inference_mode_correctness(self):
        result = torch.autograd.inference_mode()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_inference_mode_large_scale(self):
        result = torch.autograd.inference_mode()
        return result


