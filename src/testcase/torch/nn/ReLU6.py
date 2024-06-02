
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ReLU6)
class TorchNNReLU6TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu6(self, input=None):
        if input is not None:
            result = torch.nn.ReLU6()(input[0])
            return [result, input]
        a = torch.randn(10)
        relu6 = torch.nn.ReLU6()
        result = relu6(a)
        return [result, [a]]

