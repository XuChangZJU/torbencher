
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConstantPad3d)
class TorchNNConstantPad3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_constant_pad3d(self):
        a = torch.randn(1, 2, 4, 4, 4)
        pad = torch.nn.ConstantPad3d(2, 3.5)
        result = pad(a)
        return result

