
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast_1d)
class TorchAtleast1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_1d_correctness(self):
        dim = random.randint(0, 10)
        tensor = torch.randn(dim)
        result = torch.atleast_1d(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_atleast_1d_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = torch.atleast_1d(tensor)
        return result

