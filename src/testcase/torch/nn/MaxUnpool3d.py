import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool3d)
class TorchNnMaxunpool3dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool3d_correctness(self):
        # Randomly generate dimensions for the input tensor
        N = random.randint(1, 4)  # Batch size
        C = random.randint(1, 4)  # Number of channels
        # Set minimum dimensions to avoid too small inputs and to simplify calculation
        min_dim = 5
        max_dim = 10
        # Initialize kernel_size and stride
        kernel_size = random.randint(2, 4)
        min_padding = 0
        max_padding = kernel_size // 2  # Padding should not exceed half of kernel_size to keep dimensions valid
        padding = random.randint(min_padding, max_padding)
        
        # Adjust min_dim to ensure after pooling we have a valid dimension (at least 1)
        min_valid_dim = (min_dim + 2*random.randint(0, 1) - kernel_size) // kernel_size + 1
        min_dim = max(kernel_size, min_valid_dim * kernel_size - 2*random.randint(0, 1))
        while True:
            D_in = random.randint(min_dim, max_dim)
            H_in = random.randint(min_dim, max_dim)
            W_in = random.randint(min_dim, max_dim)
            
            # Check if dimensions with chosen padding can be divided by kernel_size without remainder
            # This also implicitly checks that padding won't lead to output dimensions of zero
            if (D_in + 2*padding - kernel_size) % kernel_size == 0 and \
               (H_in + 2*padding - kernel_size) % kernel_size == 0 and \
               (W_in + 2*padding - kernel_size) % kernel_size == 0:
                break
    
        stride = kernel_size  # To ensure valid unpooling
    
        # Create random input tensor
        input_tensor = torch.randn(N, C, D_in, H_in, W_in)
    
        # Create MaxPool3d and MaxUnpool3d layers
        pool = torch.nn.MaxPool3d(kernel_size, stride=stride, padding=padding, return_indices=True)
        unpool = torch.nn.MaxUnpool3d(kernel_size, stride=stride, padding=padding)
    
        # Perform max pooling
        pooled_output, indices = pool(input_tensor)
    
        # Perform max unpooling
        unpooled_output = unpool(pooled_output, indices)
    
        return unpooled_output
    
    
    
    