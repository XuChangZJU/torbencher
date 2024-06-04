
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bilinear)
class TorchBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear(self):
        input1 = torch.randn(100, 10)
        input2 = torch.randn(100, 20)
        weight = torch.randn(30, 10, 20)
        bias = torch.randn(30)
        result = torch.bilinear(input1, input2, weight, bias)
        return result

