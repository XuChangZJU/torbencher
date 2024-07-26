import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.arccos)
class TorchTensorArccosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_arccos_correctness(self):
        # Generate a random dimension for the tensor
        dim = random.randint(1, 4)  
        # Generate a random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5) 
        # Create the input size list
        input_size = [num_of_elements_each_dim for i in range(dim)] 
        # Generate a random tensor with values between -1 and 1
        input_tensor = torch.rand(input_size) * 2 - 1 
        # Calculate the arccos of the input tensor
        result = input_tensor.arccos()
        return result
    
    
    
    