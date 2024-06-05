
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vecdot)
class TorchLinalgVecdotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_vecdot_correctness(self):
        dim = random.randint(2, 10)
        x = torch.randn(dim)
        y = torch.randn(dim)
        result = torch.linalg.vecdot(x, y)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_vecdot_large_scale(self):
        dim = random.randint(100, 1000)
        x = torch.randn(dim)
        y = torch.randn(dim)
        result = torch.linalg.vecdot(x, y)
        return result

