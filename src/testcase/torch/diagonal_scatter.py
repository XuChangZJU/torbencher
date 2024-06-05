
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diagonal_scatter)
class TorchDiagonalScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal_scatter_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10))
        src = torch.randn(random.randint(1, 10))
        dim1 = random.randint(0, 1)
        dim2 = random.randint(0, 1)
        result = torch.diagonal_scatter(input, src, dim1, dim2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_diagonal_scatter_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        src = torch.randn(random.randint(1000, 10000))
        dim1 = random.randint(0, 1)
        dim2 = random.randint(0, 1)
        result = torch.diagonal_scatter(input, src, dim1, dim2)
        return result

