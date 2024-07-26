import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveMaxPool2d)
class TorchNnAdaptivemaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_AdaptiveMaxPool2d_correctness(self):
        # Random input size
        batch_size = random.randint(1, 3)
        num_channels = random.randint(1, 3)
        input_height = random.randint(5, 10)
        input_width = random.randint(5, 10)
        input_size = (batch_size, num_channels, input_height, input_width)
    
        # Random output size
        output_height = random.randint(1, input_height)
        output_width = random.randint(1, input_width)
        output_size = (output_height, output_width)
    
        # Input tensor
        input_tensor = torch.randn(input_size)
    
        # AdaptiveMaxPool2d module
        adaptive_max_pool_2d = torch.nn.AdaptiveMaxPool2d(output_size)
    
        # Output tensor
        output_tensor = adaptive_max_pool_2d(input_tensor)
        
        return output_tensor
    
    
    
    