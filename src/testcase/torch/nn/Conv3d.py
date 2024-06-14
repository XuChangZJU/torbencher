import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Conv3d)
class TorchNnConv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv3d_correctness(self):
        # Randomly generate input dimensions
        N = random.randint(1, 4)  # Batch size
        C_in = random.randint(1, 4)  # Number of input channels
        D_in = random.randint(5, 10)  # Depth of input
        H_in = random.randint(5, 10)  # Height of input
        W_in = random.randint(5, 10)  # Width of input
    
        # Randomly generate Conv3d parameters
        C_out = random.randint(1, 4)  # Number of output channels
        kernel_size = random.randint(1, 3)  # Kernel size (same for depth, height, width)
        stride = random.randint(1, 2)  # Stride (same for depth, height, width)
        padding = random.randint(0, 1)  # Padding (same for depth, height, width)
        dilation = random.randint(1, 2)  # Dilation (same for depth, height, width)
    
        # Create random input tensor
        input_tensor = torch.randn(N, C_in, D_in, H_in, W_in)
    
        # Initialize Conv3d layer
        conv3d_layer = torch.nn.Conv3d(C_in, C_out, kernel_size, stride, padding, dilation)
    
        # Apply Conv3d layer to input tensor
        output_tensor = conv3d_layer(input_tensor)
        return output_tensor
    
    
    
    