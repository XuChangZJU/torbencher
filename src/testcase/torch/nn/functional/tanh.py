
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.tanh)
class TorchNNFunctionalTanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tanh_2d(self):
        a = torch.randn(3, 2)
        result = torch.nn.functional.tanh(a)
        return result


