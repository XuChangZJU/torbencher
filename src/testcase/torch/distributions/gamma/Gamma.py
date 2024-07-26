import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.gamma.Gamma)
class TorchDistributionsGammaGammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_gamma_Gamma_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Random concentration parameter
        concentration = torch.randn(input_size).abs()  # concentration > 0
        # Random rate parameter
        rate = torch.randn(input_size).abs()  # rate > 0

        # Create a Gamma distribution
        gamma_distribution = torch.distributions.gamma.Gamma(concentration, rate)

        # Sample from the Gamma distribution
        result = gamma_distribution.sample()

        return result
