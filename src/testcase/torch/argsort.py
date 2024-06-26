
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.argsort)
class TorchArgsortTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_argsort(self, input=None):
        if input is not None:
            result = torch.argsort(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.argsort(a, dim=1, descending=True)
        return [result, [a, 1, True]]

