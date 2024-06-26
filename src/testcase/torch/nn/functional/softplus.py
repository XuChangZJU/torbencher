
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.softplus)
class TorchNNFunctionalSoftplusTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softplus_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.softplus(input[0], beta=input[1], threshold=input[2])
            return [result, input]
        a = torch.randn(4)
        b = 1
        c = 20
        result = torch.nn.functional.softplus(a, beta=b, threshold=c)
        return [result, [a, b, c]]


