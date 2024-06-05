
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.DoubleTensor)
class TorchDoubleTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_doubletensor_correctness(self):
        dim = random.randint(1, 10)
        result = torch.randn(dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_doubletensor_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.randn(dim)
        return result

