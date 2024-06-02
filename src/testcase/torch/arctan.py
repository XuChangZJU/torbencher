
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.arctan)
class TorchArctanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arctan(self, input=None):
        if input is not None:
            result = torch.arctan(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.arctan(a)
        return [result, [a]]


