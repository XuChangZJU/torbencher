import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GELU)
class TorchNNGELUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gelu(self, input=None):
        if input is not None:
            result = torch.nn.GELU()(input[0])
            return [result, input]
        a = torch.randn(10)
        gelu = torch.nn.GELU()
        result = gelu(a)
        return [result, [a]]

