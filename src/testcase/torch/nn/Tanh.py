
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Tanh)
class TorchNNTanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tanh(self, input=None):
        if input is not None:
            result = torch.nn.Tanh()(input[0])
            return result
        a = torch.randn(10)
        tanh = torch.nn.Tanh()
        result = tanh(a)
        return result

