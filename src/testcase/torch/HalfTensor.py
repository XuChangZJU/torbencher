
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.HalfTensor)
class TorchHalfTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_halftensor_correctness(self):
        dim = random.randint(1, 10)
        result = torch.randn(dim).to(torch.half)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_halftensor_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.randn(dim).to(torch.half)
        return result

