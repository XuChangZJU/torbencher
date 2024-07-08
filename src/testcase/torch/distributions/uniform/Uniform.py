import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.uniform.Uniform)
class TorchDistributionsUniformUniformTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_uniform_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random low and high values for the uniform distribution
        low = torch.randn(input_size)
        high = low + torch.rand(input_size) * 10  # Ensure high > low
        # Create a Uniform distribution
        uniform_distribution = torch.distributions.uniform.Uniform(low, high)
        # Sample from the distribution
        result = uniform_distribution.sample()
        return result
    