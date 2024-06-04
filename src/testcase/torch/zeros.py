
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.zeros)
class TorchZerosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zeros(self):
        
        a = (2, 3)
        result = torch.zeros(a)
        return result
