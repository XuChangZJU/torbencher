
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pairwise_distance)
class PairwiseDistanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pairwise_distance_correctness(self):
        input1 = torch.randn(10, 10)
        input2 = torch.randn(10, 10)
        p = random.randint(1, 5)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.pairwise_distance(input1, input2, p, eps)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_pairwise_distance_large_scale(self):
        input1 = torch.randn(1000, 1000)
        input2 = torch.randn(1000, 1000)
        p = random.randint(1, 5)
        eps = random.uniform(0.0, 1.0)
        result = torch.nn.functional.pairwise_distance(input1, input2, p, eps)
        return result

