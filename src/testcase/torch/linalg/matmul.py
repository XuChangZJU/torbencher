
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matmul)
class TorchLinalgMatmulTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_matmul_correctness(self):
        m = random.randint(2, 10)
        n = random.randint(2, 10)
        p = random.randint(2, 10)
        A = torch.randn(m, n)
        B = torch.randn(n, p)
        result = torch.linalg.matmul(A, B)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_matmul_large_scale(self):
        m = random.randint(100, 1000)
        n = random.randint(100, 1000)
        p = random.randint(100, 1000)
        A = torch.randn(m, n)
        B = torch.randn(n, p)
        result = torch.linalg.matmul(A, B)
        return result

