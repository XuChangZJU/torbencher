import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.PixelShuffle)
class TorchNnPixelshuffleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_pixelshuffle_correctness(self):
        # Random upscale factor
        upscale_factor = random.randint(2, 5) # upscale_factor should be at least 2
    
        # Random input size
        batch_size = random.randint(1, 3)
        num_channels = random.randint(1, 3) * upscale_factor * upscale_factor # num_channels should be divisible by upscale_factor^2
        height = random.randint(1, 10)
        width = random.randint(1, 10)
    
        input_size = [batch_size, num_channels, height, width]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Create PixelShuffle module
        pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
    
        # Apply PixelShuffle operation
        output_tensor = pixel_shuffle(input_tensor)
    
        return output_tensor
    