import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyConv1d)
class TorchNnLazyconv1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazyconv1d_correctness(self):
        # Randomly generate parameters for LazyConv1d
        out_channels = random.randint(1, 10)  # Number of output channels
        kernel_size = random.randint(1, 5)  # Size of the convolving kernel
        stride = random.randint(1, 3)  # Stride of the convolution
        padding = random.randint(0, 2)  # Zero-padding added to both sides of the input
        dilation = random.randint(1, 3)  # Spacing between kernel elements
        groups = random.randint(1, 3)  # Number of blocked connections from input channels to output channels
        bias = random.choice([True, False])  # Whether to add a learnable bias to the output
    
        # Randomly generate input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        in_channels = random.randint(1, 10)  # Random number of input channels
        input_length = random.randint(10, 20)  # Random length of the input sequence
        input_tensor = torch.randn(batch_size, in_channels, input_length)
    
        # Initialize LazyConv1d layer
        lazy_conv1d = torch.nn.LazyConv1d(out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
        # Apply LazyConv1d to input tensor
        result = lazy_conv1d(input_tensor)
        return result
    
    
    
    