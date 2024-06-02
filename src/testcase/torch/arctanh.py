
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arctanh)
class TorchArctanhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctanh(self, input=None):
        if input is not None:
            result = torch.arctanh(input[0])
            return [result, input]
        a = torch.randn(4).uniform_(-1, 1)
        result = torch.arctanh(a)
        return [result, [a]]


