
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.concat)
class TorchConcatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_concat_correctness(self):
        tensors = [torch.randn(random.randint(1, 10), random.randint(1, 10)) for _ in range(random.randint(1, 10))]
        dim = random.randint(0, 1)
        result = torch.concat(tensors, dim=dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_concat_large_scale(self):
        tensors = [torch.randn(random.randint(1000, 10000), random.randint(1000, 10000)) for _ in range(random.randint(1000, 10000))]
        dim = random.randint(0, 1)
        result = torch.concat(tensors, dim=dim)
        return result

