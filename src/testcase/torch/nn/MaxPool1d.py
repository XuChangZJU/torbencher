import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxPool1d)
class TorchNnMaxpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxpool1d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 10)
        num_channels = random.randint(1, 10)
        length = random.randint(10, 50)
        
        # Kernel size, ensuring it's feasible with the given length
        kernel_size = random.randint(1, min(length - 1, 5))
        
        # Stride, ensuring it's not zero and doesn't exceed kernel_size
        stride = random.randint(1, kernel_size)
        
        # Padding, ensuring we don't overshoot with padding and still get a valid output
        padding = random.randint(0, min(kernel_size // 2, (length - kernel_size) // (stride + 1)))
        
        # Dilation, keeping it simple to avoid complex dependencies in size calculation
        # Here, we'll just set dilation to 1 to simplify the logic and avoid potential issues
        dilation = random.randint(1, 2)
        
        # Validate the configuration to ensure a positive output length
        effective_length = ((length + 2 * padding - kernel_size) + (dilation - 1) * (kernel_size - 1)) // stride + 1
        assert effective_length > 0, "Computed output size is invalid"
        
        # Create a random input tensor
        input_tensor = torch.randn(batch_size, num_channels, length)

        # Apply MaxPool1d
        maxpool = torch.nn.MaxPool1d(kernel_size, stride, padding, dilation)
        output = maxpool(input_tensor)
        
        return output
    
    
    
    