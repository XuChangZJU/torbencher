import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.distribution.Distribution)
class TorchDistributionsDistributionDistributionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_distribution_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random tensor
        tensor = torch.randn(input_size)
        # Create a distribution object (e.g., Normal distribution)
        distribution = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        # Calculate the log probability of the tensor
        result = distribution.log_prob(tensor)
        return result
