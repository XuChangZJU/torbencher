import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_zero_)
class TorchForeachzeroTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_zero_correctness(self):
        # foreach_zero_ operator applies to a list of tensors.
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5)
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        num_of_tensor = random.randint(1, 3) # Randomly generate number of tensors
        tensor_list = [torch.randn(input_size) for i in range(num_of_tensor)]
    
        torch._foreach_zero_(tensor_list)
        return tensor_list
    
    
    
    
    
    
    