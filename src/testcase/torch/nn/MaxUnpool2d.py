import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool2d)
class TorchNnMaxunpool2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool2d_correctness(self):
        # Randomly generate dimensions for kernel size ensuring compatibility
        kernel_size = random.choice([2, 3])  # Limited choices to maintain validity
        # Randomly generate dimensions for the input tensor ensuring valid unpooling
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        
        # Ensure H_in and W_in can be divided by kernel_size to avoid invalid indices after pooling
        min_dim = 4  # Minimum dimension to avoid too small inputs causing issues
        while True:
            H_in = random.randint(min_dim, 8)  
            W_in = random.randint(min_dim, 8)
            if H_in % kernel_size == 0 and W_in % kernel_size == 0:
                break
        
        # Kernel size and stride, ensuring they're compatible with the input dimensions
        stride = kernel_size  # To ensure valid pooling and unpooling
    
        # Create random input tensor
        input_tensor = torch.randn(N, C, H_in, W_in)
    
        # Create MaxPool2d and MaxUnpool2d layers
        pool = torch.nn.MaxPool2d(kernel_size, stride=stride, return_indices=True)
        unpool = torch.nn.MaxUnpool2d(kernel_size, stride=stride)
    
        # Perform max pooling
        pooled_output, indices = pool(input_tensor)
    
        # Perform max unpooling, now with a guarantee that indices are valid
        unpooled_output = unpool(pooled_output, indices)
    
        return unpooled_output
    
    
    
    