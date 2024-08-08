import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.weibull.Weibull)
class TorchDistributionsWeibullWeibullTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_weibull_correctness_small_scale(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random scale parameter (lambda) between 0.1 and 1.0
        scale = torch.rand(input_size) * 0.9 + 0.1
        # Random concentration parameter (k/shape)
        concentration = torch.rand(input_size)
        # Create a Weibull distribution
        weibull_distribution = torch.distributions.weibull.Weibull(scale, concentration)
        # Sample from the Weibull distribution
        result = weibull_distribution.sample()
        return result
