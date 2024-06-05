
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.multi_dot)
class TorchLinalgMultiDotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_multi_dot_correctness(self):
        m = random.randint(2, 10)
        n = random.randint(2, 10)
        p = random.randint(2, 10)
        A = torch.randn(m, n)
        B = torch.randn(n, p)
        C = torch.randn(p, m)
        result = torch.linalg.multi_dot([A, B, C])
        return result

    @test_api_version.larger_than("1.8.0")
    def test_multi_dot_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        p = random.randint(100, 1000)
        A = torch.randn(m, n)
        B = torch.randn(n, p)
        C = torch.randn(p, m)
        result = torch.linalg.multi_dot([A, B, C])
        return result

