import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_sinh_)
class TorchForeachsinhTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_sinh_correctness(self):
        # foreach_sinh_ is an inplace function, so we test its correctness by comparing the result with torch.sinh
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        tensor_list = [torch.randn(input_size), torch.randn(input_size), torch.randn(input_size)]
        tensor_list_copy = [tensor.clone() for tensor in tensor_list]
        
        torch._foreach_sinh_(tensor_list)
        result = tensor_list[0]
        
        for i in range(len(tensor_list_copy)):
            tensor_list_copy[i] = torch.sinh(tensor_list_copy[i])
        
        return result
    
    
    
    
    
    
    