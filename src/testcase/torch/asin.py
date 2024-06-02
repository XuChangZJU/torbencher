
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.asin)
class TorchAsinTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_asin(self, input=None):
        if input is not None:
            result = torch.asin(input[0])
            return [result, input]
        a = torch.rand(4)
        result = torch.asin(a)
        return [result, [a]]

