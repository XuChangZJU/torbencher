import torch
import random
from scipy.stats import norm
import math


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.functional.gelu)
class TorchNnFunctionalGeluTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gelu_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        result = torch.nn.functional.gelu(input_tensor)
        return result
    
    def test_gelu_tanh_approximation(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        input_tensor = torch.randn(input_size)
        result = torch.nn.functional.gelu(input_tensor, approximate='tanh')
        return result
    
    