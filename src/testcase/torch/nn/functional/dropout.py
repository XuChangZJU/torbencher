
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.dropout)
class TorchNNFunctionalDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_dropout_common(self):
        a = torch.randn(1, 1, 1, 1)
        b = 0.5
        c = True
        d = False
        result = torch.nn.functional.dropout(a, b, training=c, inplace=d)
        return result


