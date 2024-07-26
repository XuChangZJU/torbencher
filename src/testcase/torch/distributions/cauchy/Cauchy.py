import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.cauchy.Cauchy)
class TorchDistributionsCauchyCauchyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_cauchy_Cauchy_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated loc, a float or Tensor
        loc = torch.randn(input_size)
        # Randomly generated scale, a float or Tensor
        scale = torch.rand(input_size) + 1e-05  # Ensure scale is positive
        # Generate Cauchy distribution
        cauchy_distribution = torch.distributions.cauchy.Cauchy(loc, scale)
        # Sample from the Cauchy distribution
        result = cauchy_distribution.sample()
        return result
