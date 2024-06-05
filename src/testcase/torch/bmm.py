
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bmm)
class TorchBmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bmm_correctness(self):
        batch = random.randint(1, 10)
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        batch1 = torch.randn(batch, m, k)
        batch2 = torch.randn(batch, k, n)
        result = torch.bmm(batch1, batch2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bmm_large_scale(self):
        batch = random.randint(100, 1000)
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        batch1 = torch.randn(batch, m, k)
        batch2 = torch.randn(batch, k, n)
        result = torch.bmm(batch1, batch2)
        return result

