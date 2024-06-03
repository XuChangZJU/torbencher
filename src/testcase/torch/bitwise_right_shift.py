
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.bitwise_right_shift)
class TorchBitwiseRightShiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_0d(self, input=None):
        if input is not None:
            result = torch.bitwise_right_shift(input[0], input[1])
            return result
        a = torch.tensor(12)
        b = 2
        result = torch.bitwise_right_shift(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_1d(self, input=None):
        if input is not None:
            result = torch.bitwise_right_shift(input[0], input[1])
            return result
        a = torch.randint(0, 10, (4,), dtype=torch.int32)
        b = 2
        result = torch.bitwise_right_shift(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_2d(self, input=None):
        if input is not None:
            result = torch.bitwise_right_shift(input[0], input[1])
            return result
        a = torch.randint(0, 10, (4, 4), dtype=torch.int32)
        b = 2
        result = torch.bitwise_right_shift(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_3d(self, input=None):
        if input is not None:
            result = torch.bitwise_right_shift(input[0], input[1])
            return result
        a = torch.randint(0, 10, (4, 4, 4), dtype=torch.int32)
        b = 2
        result = torch.bitwise_right_shift(a, b)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_bitwise_right_shift_4d(self, input=None):
        if input is not None:
            result = torch.bitwise_right_shift(input[0], input[1])
            return result
        a = torch.randint(0, 10, (4, 4, 4, 4), dtype=torch.int32)
        b = 2
        result = torch.bitwise_right_shift(a, b)
        return result


