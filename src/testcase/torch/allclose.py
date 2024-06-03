
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.allclose)
class TorchAllcloseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_allclose(self, input=None):
        if input is not None:
            result = torch.allclose(input[0], input[1])
            return result
        a = torch.tensor([10000., 1e-07])
        b = torch.tensor([10000.1, 1e-08])
        result = torch.allclose(a, b, rtol=1e-05, atol=1e-08)
        return result

