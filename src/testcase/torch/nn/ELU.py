
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ELU)
class TorchNNELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_elu(self):
        
        a = torch.randn(10)
        elu = torch.nn.ELU()
        result = elu(a)
        return result

