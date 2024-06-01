import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addbmm)
class TorchAddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addbmm_4d(self, input=None):
        if input is not None:
            result = torch.addbmm(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4, 6)
        b = torch.randn(4, 5, 6)
        c = torch.randn(4, 5, 6)
        result = torch.addbmm(a, b, c)
        return [result, [a, b, c]]

