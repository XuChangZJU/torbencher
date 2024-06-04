
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.rrelu)
class TorchNNFunctionalRReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rrelu_common(self):
        
        a = torch.randn(2, 4)
        b = 1.0 / 8
        c = 1.0 / 3
        d = False
        e = False
        result = torch.nn.functional.rrelu(a, lower=b, upper=c, training=d, inplace=e)
        return result


