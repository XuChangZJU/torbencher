import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.quantile)
class TorchTensorQuantileTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_quantile_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        q = random.uniform(0.0, 1.0)  # Random q value between 0.0 and 1.0
        dim = random.randint(0, dim - 1)  # Random dim value between 0 and dim-1
        result = input_tensor.quantile(q, dim)
        return result
    
    
    
    