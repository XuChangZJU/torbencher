
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.where)
class TorchWhereTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_where_correctness(self):
        condition = torch.randint(0, 2, (random.randint(1, 10), random.randint(1, 10)), dtype=torch.bool)
        input = torch.randn(random.randint(1, 10), random.randint(1, 10))
        other = torch.randn(random.randint(1, 10), random.randint(1, 10))
        result = torch.where(condition, input, other)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_where_large_scale(self):
        condition = torch.randint(0, 2, (random.randint(1000, 10000), random.randint(1000, 10000)), dtype=torch.bool)
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        other = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        result = torch.where(condition, input, other)
        return result

