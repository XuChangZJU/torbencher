
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Linear)
class TorchNNLinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_linear(self):
        
        a = torch.randn(10, 5)
        linear = torch.nn.Linear(5, 2)
        result = linear(a)
        return result

