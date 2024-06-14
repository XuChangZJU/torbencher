import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PixelUnshuffle)
class TorchNnPixelunshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixel_unshuffle_correctness(self):
        # Randomly generate input tensor size
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 5)
        downscale_factor = random.randint(2, 5) # downscale_factor should be at least 2
        height = random.randint(1, 10) * downscale_factor # height should be divisible by downscale_factor
        width = random.randint(1, 10) * downscale_factor # width should be divisible by downscale_factor
        input_size = [batch_size, channels, height, width]
    
        input_tensor = torch.randn(input_size)
        pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor)
        result = pixel_unshuffle(input_tensor)
        return result
    
    
    
    