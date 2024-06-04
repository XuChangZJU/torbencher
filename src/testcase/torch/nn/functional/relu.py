
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.relu)
class TorchNNFunctionalReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu_common(self):
        
        a = torch.randn(2, 4)
        b = False
        result = torch.nn.functional.relu(a, inplace=b)
        return result


