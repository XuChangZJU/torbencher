import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.init.uniform_)
class TorchNnInitUniformTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_uniform_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with the specified input size
        tensor = torch.randn(input_size)
        # Generate random lower bound and upper bound for the uniform distribution
        a = random.uniform(-10.0, 10.0)
        b = random.uniform(a, 10.0)  # Ensure b is greater than a
        # Apply uniform initialization to the tensor
        result = torch.nn.init.uniform_(tensor, a, b)
        return result
    
    
    
    