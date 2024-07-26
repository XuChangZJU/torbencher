import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.exponential.Exponential)
class TorchDistributionsExponentialExponentialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_exponential_Exponential_correctness(self):
        # Generate random input size
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random rate
        rate = torch.randn(input_size).abs()  # rate should be positive

        # Create Exponential distribution
        exponential_distribution = torch.distributions.exponential.Exponential(rate)

        # Sample from the distribution
        result = exponential_distribution.sample()
        return result
