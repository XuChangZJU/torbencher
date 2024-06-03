
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.celu)
class TorchNNFunctionalCELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_celu_common(self, input=None):
        if input is not None:
            result = torch.nn.functional.celu(input[0], input[1], inplace=input[2])
            return result
        a = torch.randn(3, 2)
        b = 1.5
        c = False
        result = torch.nn.functional.celu(a, b, inplace=c)
        return result


