
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.imag)
class TorchImagTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_imag(self):
        a = torch.randn(4, dtype=torch.cfloat)
        result = torch.imag(a)
        return result


