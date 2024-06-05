
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addbmm)
class TorchAddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addbmm_correctness(self):
        batch = random.randint(1, 10)
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        input = torch.randn(batch, m, n)
        batch1 = torch.randn(batch, m, k)
        batch2 = torch.randn(batch, k, n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addbmm(input, batch1, batch2, beta=beta)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_addbmm_large_scale(self):
        batch = random.randint(100, 1000)
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        input = torch.randn(batch, m, n)
        batch1 = torch.randn(batch, m, k)
        batch2 = torch.randn(batch, k, n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addbmm(input, batch1, batch2, beta=beta)
        return result

