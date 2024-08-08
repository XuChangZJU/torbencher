import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli)
class TorchDistributionsRelaxedUbernoulliLogitrelaxedbernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_logit_relaxed_bernoulli_correctness(self):
        # Define the parameters for the LogitRelaxedBernoulli distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        temperature = torch.rand(input_size) + 1e-5  # temperature should be positive
        probs = torch.rand(input_size)  # probs should be in range [0, 1]

        # Create a LogitRelaxedBernoulli distribution
        logit_relaxed_bernoulli_distribution = torch.distributions.relaxed_bernoulli.LogitRelaxedBernoulli(temperature,
                                                                                                           probs=probs)

        # Sample from the distribution
        samples = logit_relaxed_bernoulli_distribution.sample()
        return samples
