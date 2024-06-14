import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.GLU)
class TorchNnGluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_glu_correctness(self):
        # Define the dimension to split the input tensor
        dim = random.randint(-3, 3) # Random dimension
        # Generate a random input tensor size
        num_of_elements_each_dim = random.randint(1, 5) # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(abs(dim) - 1)] # Generate input size
        input_size.insert(abs(dim)-1, num_of_elements_each_dim * 2) # Insert the dimension to be split
    
        # Create a random input tensor
        input_tensor = torch.randn(input_size)
    
        # Apply the GLU operation
        glu_result = torch.nn.GLU()(input_tensor, dim)
    
        return glu_result
    
    
    
    