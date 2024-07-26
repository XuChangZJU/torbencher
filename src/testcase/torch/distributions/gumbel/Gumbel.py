import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.gumbel.Gumbel)
class TorchDistributionsGumbelGumbelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_gumbel_Gumbel_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated loc parameter
        loc = torch.randn(input_size)
        # Randomly generated scale parameter
        scale = torch.randn(input_size).abs()  # scale should be positive
        # Generate Gumbel distribution
        gumbel_distribution = torch.distributions.gumbel.Gumbel(loc, scale)
        # Sample from the Gumbel distribution
        result = gumbel_distribution.sample()
        return result
