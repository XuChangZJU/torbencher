
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.empty)
class TorchEmptyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_empty(self, input=None):
        if input is not None:
            result = torch.empty(input[0])
            return [result, input]
        result = torch.empty(4)
        return [result, [(4,)]]

