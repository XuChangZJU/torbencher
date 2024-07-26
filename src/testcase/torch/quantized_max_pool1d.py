import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.quantized_max_pool1d)
class TorchQuantizedmaxpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_quantized_max_pool1d_correctness(self):
        # Define the dimensions for the input tensor
        batch_size = random.randint(1, 3)
        channels = random.randint(1, 3)
        length = random.randint(3, 10)  # Length should be at least 3 for a meaningful test
        input_size = [batch_size, channels, length]
    
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
    
        # Quantize the input tensor
        scale = random.uniform(0.1, 1.0)
        zero_point = random.randint(0, 10)
        quantized_input_tensor = torch.quantize_per_tensor(input_tensor, scale, zero_point, torch.quint8)
    
        # Generate random kernel size
        kernel_size = random.randint(1, length // 2)  # Ensure kernel size is valid
    
        # Apply quantized max pooling
        result = torch.quantized_max_pool1d(quantized_input_tensor, kernel_size)
        return result
    
    
    
    
    
    
    