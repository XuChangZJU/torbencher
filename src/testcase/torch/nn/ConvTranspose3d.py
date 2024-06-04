
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConvTranspose3d)
class TorchNNConvTranspose3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose3d(self):
        a = torch.randn(1, 2, 4, 4, 4)
        conv = torch.nn.ConvTranspose3d(2, 4, 3)
        result = conv(a)
        return result

