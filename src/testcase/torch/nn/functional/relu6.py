
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.relu6)
class TorchNNFunctionalReLU6TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu6_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.relu6(input[0], inplace=input[1])
            return result
        a = torch.randn(2, 4)
        b = False
        result = torch.nn.functional.relu6(a, inplace=b)
        return result


