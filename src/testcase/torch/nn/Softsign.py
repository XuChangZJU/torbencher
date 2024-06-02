
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Softsign)
class TorchNNSoftsignTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_softsign(self, input=None):
        if input is not None:
            result = torch.nn.Softsign()(input[0])
            return [result, input]
        a = torch.randn(10)
        softsign = torch.nn.Softsign()
        result = softsign(a)
        return [result, [a]]

