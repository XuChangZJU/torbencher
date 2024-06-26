
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.prelu)
class TorchNNFunctionalPReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_prelu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.prelu(input[0], input[1])
            return [result, input]
        a = torch.randn(2, 4)
        b = torch.tensor([0.1, -0.2])
        result = torch.nn.functional.prelu(a, b)
        return [result, [a, b]]


