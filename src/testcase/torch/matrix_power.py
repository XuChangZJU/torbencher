
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.matrix_power)
class TorchMatrixPowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_matrix_power_4d(self, input=None):
        if input is not None:
            result = torch.matrix_power(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.matrix_power(a, 3)
        return [result, [a, 3]]

