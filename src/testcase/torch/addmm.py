
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmm)
class TorchAddmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmm(self):
        input1 = torch.randn(2, 3)
        input2 = torch.randn(3, 4)
        input3 = torch.randn(2, 4)
        result = torch.addmm(input3, input1, input2)
        return result

