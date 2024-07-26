import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveAvgPool1d)
class TorchNnAdaptiveavgpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_avg_pool1d_correctness(self):
        # Random input size
        num_of_channels = random.randint(1, 10)  
        input_length = random.randint(1, 10)
        batch_size = random.randint(1, 5)
        output_size = random.randint(1, input_length) # output_size should be less than or equal to input_length
    
        # Generate random input tensor
        input_tensor = torch.randn(batch_size, num_of_channels, input_length)
    
        # Create AdaptiveAvgPool1d module
        adaptive_avg_pool_1d = torch.nn.AdaptiveAvgPool1d(output_size)
    
        # Perform adaptive average pooling
        output_tensor = adaptive_avg_pool_1d(input_tensor)
        
        return output_tensor
    
    
    
    