import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.inverse_gamma.InverseGamma)
class TorchDistributionsInverseUgammaInversegammaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_inverse_gamma_InverseGamma_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Randomly generated concentration, must be positive
        concentration = torch.rand(input_size) + 1e-5
        # Randomly generated rate, must be positive
        rate = torch.rand(input_size) + 1e-5
        # Create an inverse gamma distribution
        inverse_gamma_distribution = torch.distributions.inverse_gamma.InverseGamma(concentration, rate)
        # Sample from the distribution
        result = inverse_gamma_distribution.sample()
        return result
