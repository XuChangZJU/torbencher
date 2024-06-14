import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptivemaxpool2d)
class TorchNnFunctionalAdaptivemaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        channels = random.randint(1, 4)  # Random number of channels
        height = random.randint(5, 10)  # Random height of the input tensor
        width = random.randint(5, 10)  # Random width of the input tensor
    
        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, channels, height, width)
    
        # Randomly generate the target output size
        output_height = random.randint(1, height)
        output_width = random.randint(1, width)
        output_size = (output_height, output_width)
    
        # Apply adaptive max pooling
        result = torch.nn.functional.adaptive_max_pool2d(input_tensor, output_size)
    
        return result
    