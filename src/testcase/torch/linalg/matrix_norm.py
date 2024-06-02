
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_norm)
class TorchLinalgMatrixNormTestCase(TorBencherTestCaseBase):
    def test_matrix_norm_4d_ord_str(self, input=None):
        if input is not None:
            result = torch.linalg.matrix_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        ord = 'fro'
        result = torch.linalg.matrix_norm(a, ord=ord)
        return [result, [a, ord]]

    def test_matrix_norm_4d_ord_float(self, input=None):
        if input is not None:
            result = torch.linalg.matrix_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        ord = 2.0
        result = torch.linalg.matrix_norm(a, ord=ord)
        return [result, [a, ord]]

