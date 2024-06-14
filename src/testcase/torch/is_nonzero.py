import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_nonzero)
class TorchIsnonzeroTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_nonzero_correctness(self):
        # is_nonzero only accept single element tensor, so dim is set to 1
        dim = 1
        num_of_elements_each_dim = 1
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        result = torch.is_nonzero(input_tensor)
        return result
    
    
    
    
    
    
    