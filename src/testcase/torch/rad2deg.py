
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.rad2deg)
class TorchRad2degTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rad2deg(self, input=None):
        if input is not None:
            result = torch.rad2deg(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.rad2deg(a)
        return [result, [a]]


