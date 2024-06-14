import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyConv3d)
class TorchNnLazyconv3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazyconv3d_correctness(self):
        # Randomly generate parameters for LazyConv3d
        out_channels = random.randint(1, 10)  # Number of output channels
        kernel_size = random.randint(1, 5)  # Size of the convolving kernel
        stride = random.randint(1, 3)  # Stride of the convolution
        padding = random.randint(0, 2)  # Zero-padding added to both sides of the input
        dilation = random.randint(1, 3)  # Spacing between kernel elements
    
        # Randomly generate input tensor dimensions
        batch_size = random.randint(1, 4)  # Batch size
        in_channels = random.randint(1, 10)  # Number of input channels
        depth = random.randint(10, 20)  # Depth of the input tensor
        height = random.randint(10, 20)  # Height of the input tensor
        width = random.randint(10, 20)  # Width of the input tensor
    
        # Create random input tensor
        input_tensor = torch.randn(batch_size, in_channels, depth, height, width)
    
        # Initialize LazyConv3d layer
        lazy_conv3d = torch.nn.LazyConv3d(out_channels, kernel_size, stride, padding, dilation)
    
        # Apply LazyConv3d to input tensor
        result = lazy_conv3d(input_tensor)
        return result
    