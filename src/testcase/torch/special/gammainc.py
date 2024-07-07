import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.special.gammainc)
class TorchSpecialGammaincTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gammainc_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)  
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random input tensors with values greater than 0
        input_tensor = torch.rand(input_size) + 1e-5 
        other_tensor = torch.rand(input_size) + 1e-5 
    
        # Calculate gammainc
        result = torch.special.gammainc(input_tensor, other_tensor)
        
        return result
    
    
    
    