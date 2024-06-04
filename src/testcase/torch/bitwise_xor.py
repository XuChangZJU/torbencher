
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_xor)
class TorchBitwiseXorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor(self):
        
        a = torch.tensor([True, True, False, False])
        b = torch.tensor([True, False, True, False])
        result = torch.bitwise_xor(a, b)
        return result


