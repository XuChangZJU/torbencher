
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addr)
class TorchAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_correctness(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        input = torch.randn(m, n)
        vec1 = torch.randn(m)
        vec2 = torch.randn(n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addr(input, vec1, vec2, beta=beta)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_addr_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        input = torch.randn(m, n)
        vec1 = torch.randn(m)
        vec2 = torch.randn(n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addr(input, vec1, vec2, beta=beta)
        return result

