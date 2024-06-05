
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmm)
class TorchAddmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmm_correctness(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        k = random.randint(1, 10)
        input = torch.randn(m, n)
        mat1 = torch.randn(m, k)
        mat2 = torch.randn(k, n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addmm(input, mat1, mat2, beta=beta)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_addmm_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        k = random.randint(100, 1000)
        input = torch.randn(m, n)
        mat1 = torch.randn(m, k)
        mat2 = torch.randn(k, n)
        beta = random.uniform(0.1, 10.0)
        result = torch.addmm(input, mat1, mat2, beta=beta)
        return result

