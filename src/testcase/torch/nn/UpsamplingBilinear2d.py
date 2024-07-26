import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.UpsamplingBilinear2d)
class TorchNnUpsamplingbilinear2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_upsampling_bilinear2d_correctness(self):
        # Randomly generate the batch size and number of channels
        batch_size = random.randint(1, 4)
        num_channels = random.randint(1, 4)
        
        # Randomly generate the height and width of the input tensor
        height = random.randint(2, 5)
        width = random.randint(2, 5)
        
        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, num_channels, height, width)
        
        # Randomly generate a scale factor for upsampling
        scale_factor = random.uniform(1.1, 3.0)
        
        # Create the UpsamplingBilinear2d module with the generated scale factor
        upsample = torch.nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        
        # Apply the upsampling to the input tensor
        result = upsample(input_tensor)
        
        return result
    
    
    
    