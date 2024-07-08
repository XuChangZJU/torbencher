import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.special.polygamma)
class TorchSpecialPolygammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_polygamma_correctness(self):
        # Randomly generate valid parameters for torch.special.polygamma
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1,5) # Random number of elements each dimension
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        n = random.randint(0, 10) # n should be non-negative integer
        input = torch.randn(input_size) 
        result = torch.special.polygamma(n, input)
        return result
    