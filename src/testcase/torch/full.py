import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.full)
class TorchFullTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_full_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        fill_value = random.uniform(0.1, 10.0)
        result = torch.full(input_size, fill_value)
        return result
    
    
    
    
    
    
    