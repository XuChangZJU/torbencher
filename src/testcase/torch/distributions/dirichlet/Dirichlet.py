import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.dirichlet.Dirichlet)
class TorchDistributionsDirichletDirichletTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_dirichlet_correctness(self):
        # Dirichlet distribution requires concentration > 0
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        concentration = torch.randn(input_size).abs() + 1e-6  # Ensure concentration is positive
        dirichlet_distribution = torch.distributions.dirichlet.Dirichlet(concentration)
        sample = dirichlet_distribution.sample()
        return sample
