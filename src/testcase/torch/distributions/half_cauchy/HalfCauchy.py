import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.half_cauchy.HalfCauchy)
class TorchDistributionsHalfUcauchyHalfcauchyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_HalfCauchy_correctness(self):
        # Random dimension for the scale tensor
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random scale tensor
        scale = torch.rand(input_size) + 1e-5  # Scale should be positive
        # Create HalfCauchy distribution
        half_cauchy_distribution = torch.distributions.half_cauchy.HalfCauchy(scale)
        # Sample from the distribution
        result = half_cauchy_distribution.sample()
        return result
