
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addmv)
class TorchAddmvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addmv(self, input=None):
        if input is not None:
            result = torch.addmv(input[0], input[1], input[2], beta=input[3], alpha=input[4])
            return [result, input]
        M = torch.randn(2)
        mat = torch.randn(2, 3)
        vec = torch.randn(3)
        result = torch.addmv(M, mat, vec, beta=10, alpha=0.5)
        return [result, [M, mat, vec, 10, 0.5]]

