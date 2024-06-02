
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.relu)
class TorchNNFunctionalReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.relu(input[0], inplace=input[1])
            return [result, input]
        a = torch.randn(2, 4)
        b = False
        result = torch.nn.functional.relu(a, inplace=b)
        return [result, [a, b]]


