
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReLU)
class TorchNNReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu(self, input=None):
        if input is not None:
            result = torch.nn.ReLU()(input[0])
            return [result, input]
        a = torch.randn(10)
        relu = torch.nn.ReLU()
        result = relu(a)
        return [result, [a]]

