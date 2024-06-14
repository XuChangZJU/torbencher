import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.AdaptiveMaxPool1d)
class TorchNnAdaptivemaxpool1dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_adaptive_max_pool_1d_correctness(self):
        # Random input size
        num_of_dimensions = random.randint(2, 3) # The number of dimensions should be 2 or 3
        num_of_elements_each_dim = random.randint(1, 5) # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(num_of_dimensions)] 
        if num_of_dimensions == 2:
            input_size = [input_size[0], input_size[1], random.randint(1, 10)] # Ensure the last dimension (length) is valid
    
        # Random input tensor
        input_tensor = torch.randn(input_size)
    
        # Random output size
        output_size = random.randint(1, input_size[-1]) # output_size should be less than or equal to the last dimension of input_size
    
        # Create AdaptiveMaxPool1d module
        adaptive_max_pool_1d = torch.nn.AdaptiveMaxPool1d(output_size)
    
        # Apply adaptive max pooling
        output_tensor = adaptive_max_pool_1d(input_tensor)
        
        return output_tensor
    