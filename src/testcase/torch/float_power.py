import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.float_power)
class TorchFloatpowerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_float_power_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        base = torch.randn(input_size)
        exponent = torch.randn(input_size) 
        result = torch.float_power(base, exponent)
        return result
    
    
    
    
    
    
    