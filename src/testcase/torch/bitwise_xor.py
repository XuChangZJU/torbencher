
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_xor)
class TorchBitwiseXorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor_correctness(self):
        dim = random.randint(1, 10)
        input1 = torch.randint(0, 256, (dim,))
        input2 = torch.randint(0, 256, (dim,))
        result = torch.bitwise_xor(input1, input2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_xor_large_scale(self):
        dim = random.randint(1000, 10000)
        input1 = torch.randint(0, 256, (dim,))
        input2 = torch.randint(0, 256, (dim,))
        result = torch.bitwise_xor(input1, input2)
        return result

