
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logspace)
class TorchLogspaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logspace_4d(self, input=None):
        if input is not None:
            result = torch.logspace(input[0], input[1], input[2])
            return result
        result = torch.logspace(0, 2, 5)
        return result

