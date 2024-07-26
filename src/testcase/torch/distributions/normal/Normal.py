import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.normal.Normal)
class TorchDistributionsNormalNormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_normal_Normal_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated loc
        loc = torch.randn(input_size)
        # Randomly generated scale, scale should be positive
        scale = torch.rand(input_size) + 1e-5
        # Generate normal distribution
        normal_distribution = torch.distributions.normal.Normal(loc, scale)
        # Sample from the distribution
        result = normal_distribution.sample()
        return result
