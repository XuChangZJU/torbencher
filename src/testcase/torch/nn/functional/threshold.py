
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.threshold)
class TorchNNFunctionalThresholdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_threshold_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.threshold(input[0], threshold=input[1], value=input[2], inplace=input[3])
            return [result, input]
        a = torch.randn(4)
        b = 0.1
        c = 20
        d = False
        result = torch.nn.functional.threshold(a, threshold=b, value=c, inplace=d)
        return [result, [a, b, c, d]]


