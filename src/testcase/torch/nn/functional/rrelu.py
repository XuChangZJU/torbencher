
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.rrelu)
class TorchNNFunctionalRReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rrelu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.rrelu(input[0], lower=input[1], upper=input[2], training=input[3], inplace=input[4])
            return [result, input]
        a = torch.randn(2, 4)
        b = 1.0 / 8
        c = 1.0 / 3
        d = False
        e = False
        result = torch.nn.functional.rrelu(a, lower=b, upper=c, training=d, inplace=e)
        return [result, [a, b, c, d, e]]


