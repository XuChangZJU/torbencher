
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.pairwise_distance)
class TorchNNFunctionalPairwiseDistanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pairwise_distance_common(self):
        
        a = torch.randn(100, 128)
        b = torch.randn(100, 128)
        c = 2.0
        d = 1e-06
        e = False
        result = torch.nn.functional.pairwise_distance(a, b, p=c, eps=d, keepdim=e)
        return result


