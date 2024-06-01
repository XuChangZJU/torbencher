import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.mm)
class TorchMmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mm_4d(self, input=None):
        if input is not None:
            result = torch.mm(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 5)
        b = torch.randn(5, 6)
        result = torch.mm(a, b)
        return [result, [a, b]]

