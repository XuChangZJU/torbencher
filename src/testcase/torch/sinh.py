
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sinh)
class TorchSinhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sinh(self, input=None):
        if input is not None:
            result = torch.sinh(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.sinh(a)
        return [result, [a]]

