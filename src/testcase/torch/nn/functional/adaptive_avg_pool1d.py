import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.adaptive_avg_pool1d)
class TorchNnFunctionalAdaptiveavgpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool1d_correctness(self):
        # Randomly generate the number of input planes (channels)
        num_planes = random.randint(1, 10)
        
        # Randomly generate the length of the input signal
        input_length = random.randint(5, 20)
        
        # Randomly generate the target output size
        output_size = random.randint(1, input_length)
        
        # Generate a random input tensor with the shape (batch_size, num_planes, input_length)
        batch_size = random.randint(1, 5)
        input_tensor = torch.randn(batch_size, num_planes, input_length)
        
        # Apply adaptive average pooling
        result = torch.nn.functional.adaptive_avg_pool1d(input_tensor, output_size)
        
        return result
    
    
    
    