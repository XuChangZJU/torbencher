
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_exp)
class TorchLinalgMatrixExpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.9.0")
    def test_matrix_exp_2d(self):
        a = torch.randn(4, 4)
        result = torch.linalg.matrix_exp(a)
        return result

