
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.all_reduce)
class TorchAllReduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_reduce_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        result = torch.distributed.all_reduce(tensor)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_all_reduce_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        result = torch.distributed.all_reduce(tensor)
        return result

