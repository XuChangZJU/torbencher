import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.MaxUnpool1d)
class TorchNnMaxunpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_maxunpool1d_correctness(self):
        # Randomly generate parameters for MaxPool1d and MaxUnpool1d
        kernel_size = random.randint(2, 4)
        stride = kernel_size  # To ensure valid pooling and unpooling
        padding = random.randint(0, 1)
        
        # Randomly generate input tensor dimensions
        N = random.randint(1, 3)  # Batch size
        C = random.randint(1, 3)  # Number of channels
        H_in = random.randint(8, 12)  # Height of the input tensor
        
        # Create random input tensor
        input_tensor = torch.randn(N, C, H_in)
        
        # Initialize MaxPool1d and MaxUnpool1d layers
        pool = torch.nn.MaxPool1d(kernel_size, stride=stride, padding=padding, return_indices=True)
        unpool = torch.nn.MaxUnpool1d(kernel_size, stride=stride, padding=padding)
        
        # Perform max pooling
        pooled_output, indices = pool(input_tensor)
        
        # Perform max unpooling
        unpooled_output = unpool(pooled_output, indices)
        
        return unpooled_output
    