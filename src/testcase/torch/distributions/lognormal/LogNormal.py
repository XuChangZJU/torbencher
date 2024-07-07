import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributions.log_normal.LogNormal)
class TorchDistributionsLognormalLognormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_log_normal_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1,5) 
        # Random input size
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Random loc tensor
        loc = torch.randn(input_size)
        # Random scale tensor, scale > 0
        scale = torch.rand(input_size) + 1e-5 
        # Create a log-normal distribution
        m = torch.distributions.log_normal.LogNormal(loc, scale)
        # Sample from the distribution
        result = m.sample()
        return result
    
    
    
    