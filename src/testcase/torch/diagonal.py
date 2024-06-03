
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.diagonal)
class TorchDiagonalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_diagonal(self, input=None):
        if input is not None:
            result = torch.diagonal(input[0], offset=input[1], dim1=input[2], dim2=input[3])
            return result
        a = torch.randn(4, 4)
        result = torch.diagonal(a, offset=0, dim1=0, dim2=1)
        return result
