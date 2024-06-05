
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.kineto_available)
class TorchKinetoAvailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_kineto_available_correctness(self):
        result = torch.profiler.kineto_available
        return result

    @test_api_version.larger_than("1.12.0")
    def test_kineto_available_large_scale(self):
        result = torch.profiler.kineto_available
        return result

