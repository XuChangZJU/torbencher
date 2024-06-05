
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_right_shift)
class TorchBitwiseRightShiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_correctness(self):
        dim = random.randint(1, 10)
        input = torch.randint(0, 256, (dim,))
        shift = random.randint(0, 10)
        result = torch.bitwise_right_shift(input, shift)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_large_scale(self):
        dim = random.randint(1000, 10000)
        input = torch.randint(0, 256, (dim,))
        shift = random.randint(0, 10)
        result = torch.bitwise_right_shift(input, shift)
        return result

