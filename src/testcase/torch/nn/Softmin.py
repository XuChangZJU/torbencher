
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softmin)
class TorchNNSoftminTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softmin(self):
        
        a = torch.randn(10)
        softmin = torch.nn.Softmin(dim=1)
        result = softmin(a)
        return result

