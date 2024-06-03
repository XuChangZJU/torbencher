
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.fractional_max_pool2d)
class TorchNNFunctionalFractionalMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool2d_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.fractional_max_pool2d(input[0], input[1], output_size=input[2], output_ratio=input[3], return_indices=input[4])
            return result
        a = torch.randn(1, 3, 8, 8)
        b = (2, 2)
        c = None
        d = (0.5, 0.5)
        e = False
        result = torch.nn.functional.fractional_max_pool2d(a, b, output_size=c, output_ratio=d, return_indices=e)
        return result


