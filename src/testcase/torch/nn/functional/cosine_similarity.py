
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.cosine_similarity)
class CosineSimilarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_correctness(self):
        x1 = torch.randn(10, 10)
        x2 = torch.randn(10, 10)
        dim = random.randint(0, 1)
        result = torch.nn.functional.cosine_similarity(x1, x2, dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_large_scale(self):
        x1 = torch.randn(1000, 1000)
        x2 = torch.randn(1000, 1000)
        dim = random.randint(0, 1)
        result = torch.nn.functional.cosine_similarity(x1, x2, dim)
        return result

