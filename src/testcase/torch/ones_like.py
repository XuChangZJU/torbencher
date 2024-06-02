
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ones_like)
class TorchOnes_likeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ones_like(self, input=None):
        if input is not None:
            result = torch.ones_like(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.ones_like(a)
        return [result, [a]]

