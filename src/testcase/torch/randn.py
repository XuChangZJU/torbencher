
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randn)
class TorchRandNTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randn(self, input=None):
        if input is not None:
            result = torch.randn(input[0])
            return [result, input]
        a = (2, 3)
        result = torch.randn(a)
        return [result, [a]]

