
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logsumexp)
class TorchLogsumexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logsumexp(self, input=None):
        if input is not None:
            result = torch.logsumexp(input[0], input[1], keepdim=input[2])
            return result
        a = torch.randn(4, 4)
        result = torch.logsumexp(a, 1, keepdim=True)
        return result

