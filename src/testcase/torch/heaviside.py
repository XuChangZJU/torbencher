import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.heaviside)
class TorchHeavisideTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_heaviside_correctness(self):
        # Define the dimension and size of the input tensors
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1, 5)  
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random input tensor 
        input_tensor = torch.randn(input_size) 
        
        # Generate random values tensor with the same size as input tensor
        values_tensor = torch.randn(input_size)  
    
        # Calculate the result using torch.heaviside
        result = torch.heaviside(input_tensor, values_tensor)
        return result
    