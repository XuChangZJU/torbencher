
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.diagonal)
class TorchLinalgDiagonalTestCase(TorBencherTestCaseBase):
    def test_diagonal_4d(self, input=None):
        if input is not None:
            result = torch.linalg.diagonal(input[0], offset=input[1], dim1=input[2], dim2=input[3])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        offset = 1
        dim1 = 2
        dim2 = 3
        result = torch.linalg.diagonal(a, offset=offset, dim1=dim1, dim2=dim2)
        return [result, [a, offset, dim1, dim2]]

