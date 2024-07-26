import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.nn.functional.pixel_unshuffle)
class TorchNnFunctionalPixelunshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_unshuffle_correctness(self):
        # Randomly generate input tensor dimensions
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 3)
        height = random.randint(1, 10)
        width = random.randint(1, 10)
        downscale_factor = random.randint(2, 5)  # Choose a downscale factor

        # Calculate the dimensions of the unshuffled tensor
        unshuffled_height = height * downscale_factor
        unshuffled_width = width * downscale_factor

        # Create a random input tensor with the calculated dimensions
        input_tensor = torch.randn(batch_size, channels, unshuffled_height, unshuffled_width)

        # Apply pixel_unshuffle
        result = torch.nn.functional.pixel_unshuffle(input_tensor, downscale_factor)

        return result
