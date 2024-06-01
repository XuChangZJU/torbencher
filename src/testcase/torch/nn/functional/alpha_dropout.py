import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.alpha_dropout)
class TorchNNFunctionalAlphaDropoutTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_alpha_dropout_4d(self, input=None):
        if input is not None:
            result = torch.nn.functional.alpha_dropout(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4)
        p = 0.5
        training = True
        result = torch.nn.functional.alpha_dropout(a, p, training)
        return [result, [a, p, training]]

