import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Bilinear)
class TorchNNBilinearTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bilinear(self, input=None):
        if input is not None:
            result = torch.nn.Bilinear(input[0], input[1], input[2])(input[3], input[4])
            return [result, input]
        a = torch.randn(10, 20)
        b = torch.randn(10, 30)
        bilinear = torch.nn.Bilinear(20, 30, 40)
        result = bilinear(a, b)
        return [result, [20, 30, 40, a, b]]

