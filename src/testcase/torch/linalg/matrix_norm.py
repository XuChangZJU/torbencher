
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_norm)
class TorchLinalgMatrixNormTestCase(TorBencherTestCaseBase):
    def test_matrix_norm_4d_ord_str(self):
        a = torch.randn(2, 2, 3, 3)
        ord = 'fro'
        result = torch.linalg.matrix_norm(a, ord=ord)
        return result

    def test_matrix_norm_4d_ord_float(self):
        a = torch.randn(2, 2, 3, 3)
        ord = 2.0
        result = torch.linalg.matrix_norm(a, ord=ord)
        return result

