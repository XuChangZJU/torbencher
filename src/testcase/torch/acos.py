
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.acos)
class TorchAcosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_acos(self, input=None):
        if input is not None:
            result = torch.acos(input[0])
            return result
        a = torch.rand(4)
        result = torch.acos(a)
        return result

