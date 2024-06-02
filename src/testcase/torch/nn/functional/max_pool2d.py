
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_pool2d)
class TorchNNFunctionalMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool2d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.max_pool2d(input[0], input[1], stride=input[2], padding=input[3], dilation=input[4], ceil_mode=input[5], return_indices=input[6])
            return [result, input]
        a = torch.randn(20, 16, 50, 32)
        b = 3
        c = 2
        d = 0
        e = 1
        f = False
        g = False
        result = torch.nn.functional.max_pool2d(a, b, stride=c, padding=d, dilation=e, ceil_mode=f, return_indices=g)
        return [result, [a, b, c, d, e, f, g]]


