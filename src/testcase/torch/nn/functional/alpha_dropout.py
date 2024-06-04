
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.alpha_dropout)
class TorchNNFunctionalAlphaDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_alpha_dropout_common(self):
        a = torch.randn(1, 1, 1, 1)
        b = 0.5
        c = True
        result = torch.nn.functional.alpha_dropout(a, b, training=c)
        return result


