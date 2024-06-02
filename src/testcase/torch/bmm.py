
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bmm)
class TorchBmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bmm(self, input=None):
        if input is not None:
            result = torch.bmm(input[0], input[1])
            return [result, input]
        a = torch.randn(10, 3, 4)
        b = torch.randn(10, 4, 5)
        result = torch.bmm(a, b)
        return [result, [a, b]]

