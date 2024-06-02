
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.t)
class TorchTTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_t(self, input=None):
        if input is not None:
            result = torch.t(input[0])
            return [result, input]
        a = torch.randn(3, 4)
        result = torch.t(a)
        return [result, [a]]

