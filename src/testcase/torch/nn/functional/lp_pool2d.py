
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.lp_pool2d)
class TorchNNFunctionalLPPool2DTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lp_pool2d(self, input=None):
        if input is not None:
            result = torch.nn.functional.lp_pool2d(
                input[0], norm_type=input[1], kernel_size=input[2]
            )
            return [result, input]
        a = torch.randn(1, 3, 8, 8)
        b = 2
        c = 2
        result = torch.nn.functional.lp_pool2d(a, norm_type=b, kernel_size=c)
        return [result, [a, b, c]]


