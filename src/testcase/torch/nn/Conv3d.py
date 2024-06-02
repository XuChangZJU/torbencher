
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Conv3d)
class TorchNNConv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv3d(self, input=None):
        if input is not None:
            result = torch.nn.Conv3d(input[0], input[1], input[2])(input[3])
            return [result, input]
        a = torch.randn(1, 2, 4, 4, 4)
        conv = torch.nn.Conv3d(2, 4, 3)
        result = conv(a)
        return [result, [2, 4, 3, a]]

