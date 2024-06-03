
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.silu)
class TorchNNFunctionalSiLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_silu(self, input=None):
        if input is not None:
            result = torch.nn.functional.silu(input[0])
            return result
        a = torch.randn(2, 3)
        result = torch.nn.functional.silu(a)
        return result


