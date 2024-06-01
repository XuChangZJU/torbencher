import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.SELU)
class TorchNNSelUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_selu(self, input=None):
        if input is not None:
            result = torch.nn.SELU()(input[0])
            return [result, input]
        a = torch.randn(10)
        selu = torch.nn.SELU()
        result = selu(a)
        return [result, [a]]

