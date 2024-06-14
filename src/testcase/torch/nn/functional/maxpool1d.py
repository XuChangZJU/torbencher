import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.maxpool1d)
class TorchNnFunctionalMaxpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_pool1d_correctness(self):
        # Randomly generate dimensions for the input tensor
        batch_size = random.randint(1, 4)  # Random batch size
        in_channels = random.randint(1, 4)  # Random number of input channels
        iW = random.randint(5, 10)  # Random width of the input signal
    
        # Randomly generate parameters for max_pool1d
        kernel_size = random.randint(2, 4)  # Random kernel size
        stride = random.randint(1, kernel_size)  # Random stride, must be <= kernel_size
        padding = random.randint(0, kernel_size // 2)  # Random padding, must be >= 0 and <= kernel_size / 2
        dilation = random.randint(1, 2)  # Random dilation, must be > 0
    
        # Generate random input tensor
        input_tensor = torch.randn(batch_size, in_channels, iW)
    
        # Apply max_pool1d
        result = torch.nn.functional.max_pool1d(input_tensor, kernel_size, stride, padding, dilation)
    
        return result
    