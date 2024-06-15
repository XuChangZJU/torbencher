import torch
import random
from math import ceil

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.LazyConv2d)
class TorchNnLazyconv2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lazyconv2d_correctness(self):
        # Randomly generate parameters for LazyConv2d
        out_channels = random.randint(1, 10)  
        kernel_size = (random.randint(1, 5), random.randint(1, 5))  # Kernel size for both dimensions
        stride = (random.randint(1, 3), random.randint(1, 3))  # Stride for both dimensions
        padding = (random.randint(0, 2), random.randint(0, 2))  # Padding for both dimensions
        dilation = (random.randint(1, 3), random.randint(1, 3))  # Dilation for both dimensions
        groups = 1  
        bias = random.choice([True, False])  

        # Ensure the input dimensions are compatible with the convolution settings
        while True:
            batch_size = random.randint(1, 4)  
            in_channels = random.randint(1, 10)  
            # Calculate minimum dimensions considering padding, stride, and dilation
            min_height = max(ceil((kernel_size[0] - 1) * dilation[0] + 1), kernel_size[0] - padding[0])
            min_width = max(ceil((kernel_size[1] - 1) * dilation[1] + 1), kernel_size[1] - padding[1])
            height = random.randint(max(min_height, 10), 20)  
            width = random.randint(max(min_width, 10), 20)  

            # Check if at least one valid position exists after convolution
            if ((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) < 1 or \
               ((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) < 1:
                continue
                
            break
        
        # Create a random input tensor with the validated dimensions
        input_tensor = torch.randn(batch_size, in_channels, height, width)
    
        # Initialize LazyConv2d with the generated parameters
        lazy_conv2d = torch.nn.LazyConv2d(out_channels, kernel_size, stride, padding, dilation, groups, bias)
    
        # Apply LazyConv2d to the input tensor
        result = lazy_conv2d(input_tensor)
        
        return result
    
    
    
    