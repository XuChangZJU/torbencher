
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.log2)
class TorchLog2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log2(self, input=None):
        if input is not None:
            result = torch.log2(input[0])
            return [result, input]
        a = torch.randn(5)
        result = torch.log2(a)
        return [result, [a]]

