
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast_2d)
class TorchAtleast2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_2d_correctness(self):
        dim1 = random.randint(0, 10)
        dim2 = random.randint(0, 10)
        tensor = torch.randn(dim1, dim2)
        result = torch.atleast_2d(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atleast_2d_large_scale(self):
        dim1 = random.randint(1000, 10000)
        dim2 = random.randint(1000, 10000)
        tensor = torch.randn(dim1, dim2)
        result = torch.atleast_2d(tensor)
        return result

