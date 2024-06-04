
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_power)
class TorchLinalgMatrixPowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_matrix_power(self):
        a = torch.randn(3, 3)
        n = 2
        result = torch.linalg.matrix_power(a, n)
        return result

