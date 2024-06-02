
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.log_softmax)
class TorchNNFunctionalLogSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_softmax_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.log_softmax(input[0], dim=input[1])
            return [result, input]
        a = torch.randn(2, 3)
        b = 1
        result = torch.nn.functional.log_softmax(a, dim=b)
        return [result, [a, b]]


