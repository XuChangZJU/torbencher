import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.max_unpool1d)
class TorchNnFunctionalMaxunpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_max_unpool1d_correctness(self):
        # Randomly generate the size of the input tensor
        batch_size = random.randint(1, 4)
        channels = random.randint(1, 4)
        length = random.randint(5, 10)  # Length should be greater than kernel_size for valid pooling
    
        # Randomly generate kernel size, stride, and padding
        kernel_size = random.randint(2, 4)
        stride = random.randint(1, kernel_size)
        padding = random.randint(0, kernel_size - 1)
    
        # Generate random input tensor and indices tensor
        input_tensor = torch.randn(batch_size, channels, length)
        indices = torch.randint(0, kernel_size, (batch_size, channels, length))
    
        # Perform max pooling
        pooled_tensor = torch.nn.functional.max_pool1d(input_tensor, kernel_size, stride, padding, return_indices=True)[0]
    
        # Calculate the output size for unpooling
        output_size = (length - 1) * stride - 2 * padding + kernel_size
    
        # Perform max unpooling
        result = torch.nn.functional.max_unpool1d(pooled_tensor, indices, kernel_size, stride, padding, output_size=output_size)
        return result
    
    
    
    