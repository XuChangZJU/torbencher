import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.igamma)
class TorchIgammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_igamma_correctness(self):
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input = torch.rand(input_size) # Input tensor with values between 0 and 1
        other = torch.rand(input_size) # Tensor with values between 0 and 1
        result = torch.igamma(input, other)
        return result
    
    
    
    
    
    
    