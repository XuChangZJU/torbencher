
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.leaky_relu)
class TorchNNFunctionalLeakyReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu_common(self):
        
        a = torch.randn(4)
        b = 0.01
        c = False
        result = torch.nn.functional.leaky_relu(a, negative_slope=b, inplace=c)
        return result


