import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool1d)
class TorchNNFunctionalAvgPool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool1d_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.avg_pool1d(input[0], input[1], input[2], input[3], input[4], input[5], input[6])
            return [result, input]
        a = torch.randn(2, 3, 24)
        kernel_size = 3
        stride = 2
        padding = 1
        ceil_mode = False
        count_include_pad = True
        divisor_override = None
        result = torch.nn.functional.avg_pool1d(a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        return [result, [a, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override]]

