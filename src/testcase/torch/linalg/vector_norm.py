
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vector_norm)
class TorchLinalgVectorNormTestCase(TorBencherTestCaseBase):
    def test_vector_norm_4d_ord_str(self):
        a = torch.randn(2, 2, 3, 3)
        ord = 'fro'
        result = torch.linalg.vector_norm(a, ord=ord)
        return result
    def test_vector_norm_4d_ord_float(self):
        a = torch.randn(2, 2, 3, 3)
        ord = 2.0
        result = torch.linalg.vector_norm(a, ord=ord)
        return result
    def test_vector_norm_4d_ord_int(self):
        a = torch.randn(2, 2, 3, 3)
        ord = 2
        result = torch.linalg.vector_norm(a, ord=ord)
        return result
