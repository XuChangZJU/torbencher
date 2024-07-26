import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.ZeroPad1d)
class TorchNnZeropad1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ZeroPad1d_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 10)
        channels = random.randint(1, 10)
        input_width = random.randint(1, 10)
        input_size = [batch_size, channels, input_width]

        # Generate random input tensor
        input_tensor = torch.randn(input_size)

        # Randomly generate padding size
        padding_left = random.randint(1, 5)
        padding_right = random.randint(1, 5)
        padding = (padding_left, padding_right)

        # Apply ZeroPad1d
        zero_pad_1d = torch.nn.ZeroPad1d(padding)
        result = zero_pad_1d(input_tensor)
        return result
