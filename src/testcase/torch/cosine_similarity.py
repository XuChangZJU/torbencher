
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cosine_similarity)
class TorchCosineSimilarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_correctness(self):
        dim = random.randint(1, 10)
        input1 = torch.randn(dim)
        input2 = torch.randn(dim)
        dim = random.randint(1, 10)
        result = torch.cosine_similarity(input1, input2, dim=dim)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cosine_similarity_large_scale(self):
        dim = random.randint(1000, 10000)
        input1 = torch.randn(dim)
        input2 = torch.randn(dim)
        dim = random.randint(1, 10)
        result = torch.cosine_similarity(input1, input2, dim=dim)
        return result

