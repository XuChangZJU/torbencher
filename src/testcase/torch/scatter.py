
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.scatter)
class TorchScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        dim = random.randint(0, 2)
        index = torch.randint(0, random.randint(1, 10), (random.randint(1, 10),))
        src = torch.randn(random.randint(1, 10), random.randint(1, 10))
        result = torch.scatter(input, dim, index, src)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_scatter_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        dim = random.randint(0, 2)
        index = torch.randint(0, random.randint(1000, 10000), (random.randint(1000, 10000),))
        src = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        result = torch.scatter(input, dim, index, src)
        return result

