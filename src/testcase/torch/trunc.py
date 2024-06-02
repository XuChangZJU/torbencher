
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.trunc)
class TorchTruncTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_trunc(self, input=None):
        if input is not None:
            result = torch.trunc(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.trunc(a)
        return [result, [a]]

