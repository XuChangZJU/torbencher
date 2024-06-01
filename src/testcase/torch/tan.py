import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.tan)
class TorchTanTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tan_4d(self, input=None):
        if input is not None:
            result = torch.tan(input[0])
            return [result, input]
        a = torch.randn(4)
        result = torch.tan(a)
        return [result, [a]]

