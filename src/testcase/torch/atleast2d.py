import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.atleast2d)
class TorchAtleast2dTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_atleast_2d_correctness(self):
        # Case 1: Input is a single tensor with zero dimensions
        dim = random.randint(0, 1)  # Random dimension for the tensor, 0 or 1
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        input_tensor = torch.randn(input_size)
        result = torch.atleast_2d(input_tensor)
    
        # Case 2: Input is a single tensor with two or more dimensions
        dim = random.randint(2, 4)  # Random dimension for the tensor, at least 2
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        input_tensor = torch.randn(input_size)
        result = torch.atleast_2d(input_tensor)
    
        # Case 3: Input is a tuple of tensors
        num_of_tensors = random.randint(2, 4) # Random number of tensors in the tuple
        tensors = []
        for i in range(num_of_tensors):
            dim = random.randint(0, 3)  # Random dimension for each tensor
            num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
            input_size=[num_of_elements_each_dim for i in range(dim)] 
            tensors.append(torch.randn(input_size))
        result = torch.atleast_2d(tuple(tensors))
        
        return result
    