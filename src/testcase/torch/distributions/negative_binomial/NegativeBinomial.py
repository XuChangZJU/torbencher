import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.negative_binomial.NegativeBinomial)
class TorchDistributionsNegativeUbinomialNegativebinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_distributions_negative_binomial_NegativeBinomial_correctness(self):
        # Generate random parameters for NegativeBinomial distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        total_count = random.uniform(0.1, 10.0)  # total_count: non-negative number of negative Bernoulli trials to stop
        probs = torch.rand(input_size) * 0.9  # probs: Event probabilities of success in the half open interval [0, 1)

        # Create NegativeBinomial distribution
        negative_binomial_distribution = torch.distributions.negative_binomial.NegativeBinomial(total_count, probs)

        # Sample from the distribution
        samples = negative_binomial_distribution.sample()

        # Return the samples
        return samples
