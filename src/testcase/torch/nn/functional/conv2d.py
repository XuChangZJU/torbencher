import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.conv2d)
class TorchNnFunctionalConv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv2d_correctness(self):
        # Define the dimensions for the input tensor
        minibatch = random.randint(1, 4)
        in_channels = random.randint(1, 4)
        iH = random.randint(5, 10)
        iW = random.randint(5, 10)
    
        # Define the dimensions for the kernel
        out_channels = random.randint(1, 4)
        kH = random.randint(1, 4)
        kW = random.randint(1, 4)
    
        # Create the input tensor
        input_tensor = torch.randn(minibatch, in_channels, iH, iW)
    
        # Create the kernel
        kernel = torch.randn(out_channels, in_channels, kH, kW)
    
        # Perform the convolution operation
        result = torch.nn.functional.conv2d(input_tensor, kernel)
    
        # Return the result
        return result
    
    
    
    