import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.topk)
class TorchTopkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_topk_correctness(self):
        dim = random.randint(0, 3)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim + 1)] 
    
        input = torch.randn(input_size)
        k = random.randint(1, input_size[dim])  # Random k value between 1 and the size of the chosen dimension
        result = torch.topk(input, k, dim)
        return result
    
    
    
    
    
    
    