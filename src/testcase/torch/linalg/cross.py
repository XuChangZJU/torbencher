
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.cross)
class TorchLinalgCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_cross_correctness(self):
        dim = random.randint(3, 10)
        a = torch.randn(dim)
        b = torch.randn(dim)
        result = torch.linalg.cross(a, b)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_cross_large_scale(self):
        dim = random.randint(100, 1000)
        a = torch.randn(dim)
        b = torch.randn(dim)
        result = torch.linalg.cross(a, b)
        return result

