import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxPool2d)
class TorchNnMaxpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxpool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 10)  # Batch size
        C = random.randint(1, 10)  # Number of channels
        H = random.randint(10, 20)  # Height of the input tensor
        W = random.randint(10, 20)  # Width of the input tensor
    
        # Randomly generate kernel size, stride, padding, and dilation
        kernel_size = (random.randint(2, 5), random.randint(2, 5))
        stride = (random.randint(1, 3), random.randint(1, 3))
        padding = (random.randint(0, 2), random.randint(0, 2))
        dilation = (random.randint(1, 2), random.randint(1, 2))
    
        # Create a random input tensor
        input_tensor = torch.randn(N, C, H, W)
    
        # Create the MaxPool2d layer with the generated parameters
        maxpool_layer = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation)
    
        # Apply the MaxPool2d layer to the input tensor
        output_tensor = maxpool_layer(input_tensor)
    
        return output_tensor
    