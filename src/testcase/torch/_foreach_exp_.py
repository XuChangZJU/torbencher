import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_exp_)
class TorchForeachexpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_exp_correctness(self):
        # foreach_exp_ requires the length of input list to be larger than 0
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        list_len = random.randint(1, 5)
    
        tensor_list = [torch.randn(input_size) for _ in range(list_len)]
        result = torch._foreach_exp_(tensor_list)
        return result
    
    
    
    
    
    
    