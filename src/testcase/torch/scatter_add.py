
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.scatter_add)
class TorchScatterAddTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_add_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10))
        dim = random.randint(0, 1)
        index = torch.randint(0, random.randint(1, 10), (random.randint(1, 10),))
        src = torch.randn(random.randint(1, 10))
        result = torch.scatter_add(input, dim, index, src)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_scatter_add_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        dim = random.randint(0, 1)
        index = torch.randint(0, random.randint(1000, 10000), (random.randint(1000, 10000),))
        src = torch.randn(random.randint(1000, 10000))
        result = torch.scatter_add(input, dim, index, src)
        return result

