
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.kthvalue)
class TorchKthvalueTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_kthvalue(self, input=None):
        if input is not None:
            result = torch.kthvalue(input[0], input[1], dim=input[2])
            return [result[0], input]
        a = torch.randn(4, 4)
        result = torch.kthvalue(a, 3, dim=1)
        return [result[0], [a, 3, 1]]

