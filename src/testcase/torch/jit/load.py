
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.load)
class TorchJitLoadTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_load_correctness(self):
        result = torch.jit.load(f"test_model_{random.randint(1, 10)}.pt")
        return result

    @test_api_version.larger_than("1.1.3")
    def test_load_large_scale(self):
        result = torch.jit.load(f"test_model_{random.randint(1, 10)}.pt")
        return result

