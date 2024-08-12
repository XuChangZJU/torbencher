import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.continuous_bernoulli.ContinuousBernoulli)
class TorchDistributionsContinuousUbernoulliContinuousbernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_ContinuousBernoulli_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # probs should be in (0, 1)
        probs = torch.rand(input_size) * 0.9 + 0.1
        continuous_bernoulli_distribution = torch.distributions.continuous_bernoulli.ContinuousBernoulli(probs)
        # Sample from the distribution
        result = continuous_bernoulli_distribution.sample()
        return result
