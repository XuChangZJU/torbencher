
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dsmm)
class TorchDsmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dsmm_correctness(self):
        m = random.randint(1, 10)
        n = random.randint(1, 10)
        p = random.randint(1, 10)
        input = torch.randn(m, n)
        mat2 = torch.randn(n, p)
        result = torch.dsmm(input, mat2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_dsmm_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        p = random.randint(100, 1000)
        input = torch.randn(m, n)
        mat2 = torch.randn(n, p)
        result = torch.dsmm(input, mat2)
        return result

