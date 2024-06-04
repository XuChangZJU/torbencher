
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PairwiseDistance)
class TorchNNPairwiseDistanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pairwise_distance(self):
        
        a = torch.randn(3, 5)
        b = torch.randn(3, 5)
        dist = torch.nn.PairwiseDistance()
        result = dist(a, b)
        return result

