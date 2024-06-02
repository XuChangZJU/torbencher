
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.linalg.matrix_power)
class TorchLinalgMatrixPowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.8.2")
    def test_matrix_power(self, input=None):
        if input is not None:
            result = torch.linalg.matrix_power(input[0], input[1])
            return [result, input]
        a = torch.randn(3, 3)
        n = 2
        result = torch.linalg.matrix_power(a, n)
        return [result, [a, n]]

