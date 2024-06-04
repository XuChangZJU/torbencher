
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.elu)
class TorchNNFunctionalELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_elu_common(self):
        a = torch.randn(4)
        b = 1.5
        c = False
        result = torch.nn.functional.elu(a, b, inplace=c)
        return result


