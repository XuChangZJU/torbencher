import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cat)
class TorchCatTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cat_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_tensors = random.randint(1, 5) # Random number of tensors to concatenate
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim + 1)] 
    
        tensors = []
        for _ in range(num_of_tensors):
            tensors.append(torch.randn(input_size))
        result = torch.cat(tensors, dim)
        return result
    
    
    
    
    
    
    