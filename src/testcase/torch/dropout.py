
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.dropout)
class TorchDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout(self, input=None):
        if input is not None:
            result = torch.dropout(input[0], p=input[1], training=input[2])
            return result
        a = torch.randn(4)
        result = torch.dropout(a, p=0.5, training=True)
        return result

