import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.laplace.Laplace)
class TorchDistributionsLaplaceLaplaceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_distributions_laplace_Laplace_correctness(self):
        # Laplace distribution parameters
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        loc = torch.randn(input_size)  # Mean of the distribution
        scale = torch.rand(input_size) + 1e-5  # Scale of the distribution (ensuring scale > 0)

        # Create a Laplace distribution
        laplace_distribution = torch.distributions.laplace.Laplace(loc, scale)

        # Sample from the distribution
        sample = laplace_distribution.sample()

        # Return the sample
        return sample
