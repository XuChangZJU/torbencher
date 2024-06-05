
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.is_tracing)
class TorchJitIsTracingTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_tracing_correctness(self):
        result = torch.jit.is_tracing()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_tracing_large_scale(self):
        result = torch.jit.is_tracing()
        return result

