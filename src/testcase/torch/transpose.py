import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.transpose)
class TorchTransposeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_transpose_4d(self, input=None):
        if input is not None:
            result = torch.transpose(input[0], 0, 1)
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.transpose(a, 0, 1)
        return [result, [a]]

