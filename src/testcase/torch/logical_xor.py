
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.logical_xor)
class TorchLogicalXorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_logical_xor(self, input=None):
        if input is not None:
            result = torch.logical_xor(input[0], input[1])
            return result
        a = torch.tensor([True, False, True])
        b = torch.tensor([False, False, True])
        result = torch.logical_xor(a, b)
        return result

