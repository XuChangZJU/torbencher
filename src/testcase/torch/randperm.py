
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.randperm)
class TorchRandPermTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_randperm(self, input=None):
        if input is not None:
            result = torch.randperm(input[0])
            return [result, input]
        result = torch.randperm(10)
        return [result, [10]]

