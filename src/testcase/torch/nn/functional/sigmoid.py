
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.sigmoid)
class TorchNNFunctionalSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sigmoid_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.sigmoid(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.nn.functional.sigmoid(a)
        return [result, [a]]


