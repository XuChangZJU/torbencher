import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.pixel_shuffle)
class TorchNnFunctionalPixelUshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_pixel_shuffle_correctness(self):
        # Randomly generate input tensor shape
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 3)
        upscale_factor = random.randint(2, 4)  # Upscale factor
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        input_size = [batch_size, channels * (upscale_factor ** 2), height, width]

        # Create random input tensor
        input_tensor = torch.randn(input_size)

        # Apply pixel_shuffle
        output_tensor = torch.nn.functional.pixel_shuffle(input_tensor, upscale_factor)

        return output_tensor
