
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cond)
class TorchCondTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cond_correctness(self):
        dim = random.randint(1, 10)
        condition = torch.randint(0, 2, (dim,))
        true_fn = lambda x: x + 1
        false_fn = lambda x: x - 1
        input = torch.randn(dim)
        result = torch.cond(condition, true_fn, false_fn, input)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cond_large_scale(self):
        dim = random.randint(1000, 10000)
        condition = torch.randint(0, 2, (dim,))
        true_fn = lambda x: x + 1
        false_fn = lambda x: x - 1
        input = torch.randn(dim)
        result = torch.cond(condition, true_fn, false_fn, input)
        return result

