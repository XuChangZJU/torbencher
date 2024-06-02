
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.log)
class TorchLogTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log(self, input=None):
        if input is not None:
            result = torch.log(input[0])
            return [result, input]
        a = torch.randn(5)
        result = torch.log(a)
        return [result, [a]]

