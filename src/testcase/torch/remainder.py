import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.remainder)
class TorchRemainderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_remainder_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        dividend = torch.randn(input_size)
        divisor = torch.randn(input_size).apply_(lambda x: x + 0.0001 if x == 0 else x) # Ensure the divisor is not zero
        result = torch.remainder(dividend, divisor)
        return result
    
    
    
    
    
    
    