import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cross)
class TorchCrossTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cross_4d(self, input=None):
        if input is not None:
            result = torch.cross(input[0], input[1])
            return [result, input]
        a = torch.randn(3)
        b = torch.randn(3)
        result = torch.cross(a, b)
        return [result, [a, b]]

