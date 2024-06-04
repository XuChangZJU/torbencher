
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.FractionalMaxPool2d)
class TorchNNFractionalMaxPool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fractional_max_pool2d(self):
        a = torch.randn(1, 2, 4, 4)
        pool = torch.nn.FractionalMaxPool2d(3)
        result = pool(a)
        return result

