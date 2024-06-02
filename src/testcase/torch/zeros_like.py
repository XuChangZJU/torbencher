
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.zeros_like)
class TorchZeros_likeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_zeros_like(self, input=None):
        if input is not None:
            result = torch.zeros_like(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.zeros_like(a)
        return [result, [a]]



