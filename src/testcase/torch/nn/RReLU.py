
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.RReLU)
class TorchNNRReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rrelu(self, input=None):
        if input is not None:
            result = torch.nn.RReLU()(input[0])
            return result
        a = torch.randn(10)
        rrelu = torch.nn.RReLU()
        result = rrelu(a)
        return result

