
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_norm.matrix_norm)
class TorchLinalgMatrixNormMatrixNormTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11.0")
    def test_matrix_norm(self, input=None):
        if input is not None:
            result = torch.linalg.matrix_norm.matrix_norm(input[0], ord=input[1])
            return [result, input]
        a = torch.randn(3, 3)
        result = torch.linalg.matrix_norm.matrix_norm(a, ord='fro')
        return [result, [a, 'fro']]

