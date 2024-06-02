
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.vector_norm)
class TorchLinalgVectorNormTestCase(TorBencherTestCaseBase):
    def test_vector_norm_4d_ord_str(self, input=None):
        if input is not None:
            result = torch.linalg.vector_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        ord = 'fro'
        result = torch.linalg.vector_norm(a, ord=ord)
        return [result, [a, ord]]
    def test_vector_norm_4d_ord_float(self, input=None):
        if input is not None:
            result = torch.linalg.vector_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        ord = 2.0
        result = torch.linalg.vector_norm(a, ord=ord)
        return [result, [a, ord]]
    def test_vector_norm_4d_ord_int(self, input=None):
        if input is not None:
            result = torch.linalg.vector_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(2, 2, 3, 3)
        ord = 2
        result = torch.linalg.vector_norm(a, ord=ord)
        return [result, [a, ord]]
