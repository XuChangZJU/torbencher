import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.avg_pool2d)
class TorchNnFunctionalAvgpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_avg_pool2d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        in_channels = random.randint(1, 4)  # Random number of input channels
        height = random.randint(5, 10)  # Random height of the input tensor
        width = random.randint(5, 10)  # Random width of the input tensor
    
        # Create a random input tensor with the generated dimensions
        input_tensor = torch.randn(batch_size, in_channels, height, width)
    
        # Randomly generate kernel size
        kernel_height = random.randint(2, 4)
        kernel_width = random.randint(2, 4)
        kernel_size = (kernel_height, kernel_width)
    
        # Randomly generate stride, ensuring it is valid
        stride_height = random.randint(1, kernel_height)
        stride_width = random.randint(1, kernel_width)
        stride = (stride_height, stride_width)
    
        # Randomly generate padding, ensuring it is at most half of the kernel size
        padding_height = random.randint(0, kernel_height // 2)
        padding_width = random.randint(0, kernel_width // 2)
        padding = (padding_height, padding_width)
    
        # Apply avg_pool2d with the generated parameters
        result = torch.nn.functional.avg_pool2d(input_tensor, kernel_size, stride, padding)
    
        return result
    
    
    
    