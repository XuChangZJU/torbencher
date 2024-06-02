
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bilinear)
class TorchBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear(self, input=None):
        if input is not None:
            result = torch.bilinear(input[0], input[1], input[2], input[3])
            return [result, input]
        input1 = torch.randn(100, 10)
        input2 = torch.randn(100, 20)
        weight = torch.randn(30, 10, 20)
        bias = torch.randn(30)
        result = torch.bilinear(input1, input2, weight, bias)
        return [result, [input1, input2, weight, bias]]

