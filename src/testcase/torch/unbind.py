
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unbind)
class TorchUnbindTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unbind(self, input=None):
        if input is not None:
            result = torch.unbind(input[0], dim=input[1])
            return [result[0], input]
        a = torch.randn(2, 3)
        result = torch.unbind(a, dim=1)
        return [result[0], [a, 1]]

