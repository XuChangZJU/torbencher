
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vector_norm)
class TorchLinalgVectorNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.0")
    def test_vector_norm_correctness(self):
        dim = random.randint(2, 10)
        x = torch.randn(dim)
        ord = random.choice([None, 1, 2, inf])
        result = torch.linalg.vector_norm(x, ord=ord)
        return result

    @test_api_version.larger_than("1.8.0")
    def test_vector_norm_large_scale(self):
        dim = random.randint(100, 1000)
        x = torch.randn(dim)
        ord = random.choice([None, 1, 2, inf])
        result = torch.linalg.vector_norm(x, ord=ord)
        return result



import torch
import random

