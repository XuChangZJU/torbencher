
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.hardsigmoid)
class TorchNNFunctionalHardSigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardsigmoid(self, input=None):
        if input is not None:
            result = torch.nn.functional.hardsigmoid(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.nn.functional.hardsigmoid(a)
        return [result, [a]]


