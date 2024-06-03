
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Hardsigmoid)
class TorchNNHardsigmoidTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hardsigmoid(self, input=None):
        if input is not None:
            result = torch.nn.Hardsigmoid()(input[0])
            return result
        a = torch.randn(10)
        hardsigmoid = torch.nn.Hardsigmoid()
        result = hardsigmoid(a)
        return result

