
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.leaky_relu)
class TorchNNFunctionalLeakyReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.leaky_relu(input[0], negative_slope=input[1], inplace=input[2])
            return [result, input]
        a = torch.randn(4)
        b = 0.01
        c = False
        result = torch.nn.functional.leaky_relu(a, negative_slope=b, inplace=c)
        return [result, [a, b, c]]


