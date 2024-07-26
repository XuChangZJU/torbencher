import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.floor_)
class TorchTensorFloorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_floor__correctness(self):
        # Generate random dimension for the tensor
        dim = random.randint(1, 4)  
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Generate input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate a random tensor 
        input_tensor = torch.randn(input_size)
        # Apply floor_() to the tensor
        input_tensor.floor_()
        # Return the tensor after applying floor_()
        return input_tensor
    
    
    
    