
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.relu)
class TorchReluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relu_4d(self):
        
        a = torch.randn(4)
        result = torch.relu(a)
        return result

