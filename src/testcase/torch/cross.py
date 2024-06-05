
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cross)
class TorchCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_correctness(self):
        dim = random.randint(1, 10)
        tensor1 = torch.randn(dim, 3)
        tensor2 = torch.randn(dim, 3)
        dim = random.randint(1, 3)
        result = torch.cross(tensor1, tensor2, dim=dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cross_large_scale(self):
        dim = random.randint(100, 1000)
        tensor1 = torch.randn(dim, 3)
        tensor2 = torch.randn(dim, 3)
        dim = random.randint(1, 3)
        result = torch.cross(tensor1, tensor2, dim=dim)
        return result

