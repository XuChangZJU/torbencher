
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.rand)
class TorchRandTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rand(self, input=None):
        if input is not None:
            result = torch.rand(input[0])
            return result
        a = (2, 3)
        result = torch.rand(a)
        return result

