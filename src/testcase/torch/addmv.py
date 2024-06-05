
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmv)
class TorchAddmvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmv_correctness(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        input = torch.randn(m)
        mat = torch.randn(m, n)
        vec = torch.randn(n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addmv(input, mat, vec, beta=beta)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_addmv_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        input = torch.randn(m)
        mat = torch.randn(m, n)
        vec = torch.randn(n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addmv(input, mat, vec, beta=beta)
        return result

