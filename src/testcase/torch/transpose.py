
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.transpose)
class TorchTransposeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transpose_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10))
        dim0 = random.randint(0, 1)
        dim1 = random.randint(0, 1)
        result = torch.transpose(input, dim0, dim1)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_transpose_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        dim0 = random.randint(0, 1)
        dim1 = random.randint(0, 1)
        result = torch.transpose(input, dim0, dim1)
        return result

