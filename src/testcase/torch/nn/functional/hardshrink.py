
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardshrink)
class TorchNNFunctionalHardshrinkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardshrink_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.hardshrink(input[0], lambd=input[1])
            return [result, input]
        a = torch.randn(4)
        b = 0.5
        result = torch.nn.functional.hardshrink(a, lambd=b)
        return [result, [a, b]]


