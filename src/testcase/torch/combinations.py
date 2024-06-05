
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.combinations)
class TorchCombinationsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_combinations_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.arange(dim)
        r = random.randint(1, dim)
        result = torch.combinations(tensor, r=r)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_combinations_large_scale(self):
        dim = random.randint(100, 1000)
        tensor = torch.arange(dim)
        r = random.randint(1, dim)
        result = torch.combinations(tensor, r=r)
        return result

