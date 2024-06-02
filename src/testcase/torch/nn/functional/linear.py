
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.linear)
class TorchNNFunctionalLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.linear(input[0], input[1], bias=input[2])
            return [result, input]
        a = torch.randn(20, 100)
        b = torch.randn(200, 100)
        c = None
        result = torch.nn.functional.linear(a, b, bias=c)
        return [result, [a, b, c]]


