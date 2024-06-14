import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.svd_lowrank)
class TorchSvdlowrankTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_svd_lowrank_correctness(self):
        dim1 = random.randint(1, 4)  # Random dimension for the tensor
        dim2 = random.randint(1, 4)  # Random dimension for the tensor
        dim3 = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim1)] + [num_of_elements_each_dim for i in range(dim2)] + [num_of_elements_each_dim for i in range(dim3)] 
    
        a = torch.randn(input_size)
        q = random.randint(1, a.size(-1)) # q should be smaller than the last dimension of a
        result = torch.svd_lowrank(a, q)
        return result
    
    
    
    
    
    
    