import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.ConvTranspose1d)
class TorchNnConvtranspose1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_conv_transpose1d_correctness(self):
        # Randomly generate parameters for ConvTranspose1d
        in_channels = random.randint(1, 10)  # Number of input channels
        out_channels = random.randint(1, 10)  # Number of output channels
        kernel_size = random.randint(1, 5)  # Size of the convolving kernel
        stride = random.randint(1, 3)  # Stride of the convolution
        padding = random.randint(0, 2)  # Padding added to both sides of the input
        output_padding = random.randint(0, 2)  # Additional size added to one side of the output shape
        dilation = random.randint(1, 2)  # Spacing between kernel elements
    
        # Randomly generate input tensor size
        batch_size = random.randint(1, 4)  # Batch size
        input_length = random.randint(5, 20)  # Length of the input sequence
        input_tensor = torch.randn(batch_size, in_channels, input_length)
    
        # Create ConvTranspose1d layer with generated parameters
        conv_transpose1d = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation=dilation
        )
    
        # Apply ConvTranspose1d to the input tensor
        result = conv_transpose1d(input_tensor)
        return result
    
    
    
    