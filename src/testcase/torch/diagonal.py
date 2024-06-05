
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diagonal)
class TorchDiagonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal_correctness(self):
        dim1 = random.randint(1, 10)
        dim2 = random.randint(1, 10)
        tensor = torch.randn(dim1, dim2)
        result = torch.diagonal(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_diagonal_large_scale(self):
        dim1 = random.randint(100, 1000)
        dim2 = random.randint(100, 1000)
        tensor = torch.randn(dim1, dim2)
        result = torch.diagonal(tensor)
        return result

