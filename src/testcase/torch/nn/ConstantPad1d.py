
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConstantPad1d)
class TorchNNConstantPad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_constant_pad1d(self):
        a = torch.randn(1, 2, 4)
        pad = torch.nn.ConstantPad1d(2, 3.5)
        result = pad(a)
        return result

