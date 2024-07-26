import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch._foreach_log2)
class TorchForeachlog2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_foreach_log2_correctness(self):
        # foreach_log2 expects a list of tensors.
        dim = random.randint(1, 4)  
        num_of_elements_each_dim = random.randint(1,5) 
        input_size=[num_of_elements_each_dim for i in range(dim)] 
        num_tensors = random.randint(1, 3) # Random number of tensors in the list
        tensor_list = [torch.randn(input_size) for i in range(num_tensors)]
        result = torch._foreach_log2(tensor_list)
        return result
    
    
    
    
    
    
    