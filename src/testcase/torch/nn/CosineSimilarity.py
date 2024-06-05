
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.CosineSimilarity)
class TorchCosineSimilarityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cosinesimilarity_correctness(self):
        input1 = torch.randn(random.randint(1, 10), random.randint(1, 10))
        input2 = torch.randn(random.randint(1, 10), random.randint(1, 10))
        cosine_similarity = torch.nn.CosineSimilarity()
        result = cosine_similarity(input1, input2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_cosinesimilarity_large_scale(self):
        input1 = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        input2 = torch.randn(random.randint(1000, 10000), random.randint(100, 1000))
        cosine_similarity = torch.nn.CosineSimilarity()
        result = cosine_similarity(input1, input2)
        return result

