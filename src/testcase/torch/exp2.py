import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.exp2)
class TorchExp2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_exp2_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)  
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5) 
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)] 
    
        # Generate a random tensor of the specified size
        input_tensor = torch.randn(input_size)  
        
        # Calculate exp2 of the input tensor
        result = torch.exp2(input_tensor)
        
        # Return the result tensor
        return result
    