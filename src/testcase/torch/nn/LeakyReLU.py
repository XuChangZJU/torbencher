
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LeakyReLU)
class TorchNNLeakyReLUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_leaky_relu(self):
        
        a = torch.randn(10)
        leaky_relu = torch.nn.LeakyReLU()
        result = leaky_relu(a)
        return result

