import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.where)
class TorchWhereTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_where_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        condition = torch.randint(0, 2, input_size, dtype=torch.bool) # Generate random boolean tensor
        input = torch.randn(input_size) 
        other = torch.randn(input_size) 
        result = torch.where(condition, input, other)
        return result
    
    
    
    
    
    
    