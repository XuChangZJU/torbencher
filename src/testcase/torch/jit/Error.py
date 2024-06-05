
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.Error)
class TorchJitErrorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_error_correctness(self):
        error = torch.jit.Error()
        result = error.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_error_large_scale(self):
        error = torch.jit.Error()
        result = error.type
        return result

