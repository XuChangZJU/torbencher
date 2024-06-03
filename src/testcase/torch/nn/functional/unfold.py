
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.unfold)
class TorchNNFunctionalUnfoldTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unfold_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.unfold(input[0], kernel_size=input[1], dilation=input[2], padding=input[3], stride=input[4])
            return result
        a = torch.randn(2, 3, 5, 5)
        b = 3
        c = 1
        d = 1
        e = 2
        result = torch.nn.functional.unfold(a, kernel_size=b, dilation=c, padding=d, stride=e)
        return result
