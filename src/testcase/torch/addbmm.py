
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addbmm)
class TorchAddbmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addbmm(self, input=None):
        if input is not None:
            result = torch.addbmm(input[0], input[1], input[2], beta=input[3], alpha=input[4])
            return [result, input]
        a = torch.randn(3, 5)
        b = torch.randn(10, 3, 4)
        c = torch.randn(10, 4, 5)
        result = torch.addbmm(a, b, c, beta = 0.2, alpha=5)
        return [result, [a, b, c, 0.2, 5]]

