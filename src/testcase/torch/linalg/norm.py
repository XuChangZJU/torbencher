
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.norm)
class TorchLinalgNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_norm_correctness(self):
        dim = random.randint(2, 10)
        A = torch.randn(dim, dim)
        ord = random.choice([None, 'fro', 1, 2, 'nuc', inf])
        result = torch.linalg.norm(A, ord=ord)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_norm_large_scale(self):
        dim = random.randint(100, 1000)
        A = torch.randn(dim, dim)
        ord = random.choice([None, 'fro', 1, 2, 'nuc', inf])
        result = torch.linalg.norm(A, ord=ord)
        return result

