import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.softmax)
class TorchSpecialSoftmaxTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_special_softmax_4d(self, input=None):
        if input is not None:
            result = torch.special.softmax(input[0], dim=input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.special.softmax(a, dim=1)
        return [result, [a, 1]]

