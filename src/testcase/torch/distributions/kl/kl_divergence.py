import random

import torch
from torch.distributions import Bernoulli

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.kl.kl_divergence)
class TorchDistributionsKlKlUdivergenceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_kl_divergence_correctness(self):
        # Bernoulli and Bernoulli
        probs_p = random.uniform(0.1, 0.9)  # Bernoulli probability
        probs_q = random.uniform(0.1, 0.9)  # Bernoulli probability
        batch_size = random.randint(1, 3)
        bernoulli_p = Bernoulli(probs=torch.tensor([probs_p] * batch_size))
        bernoulli_q = Bernoulli(probs=torch.tensor([probs_q] * batch_size))
        result = torch.distributions.kl.kl_divergence(bernoulli_p, bernoulli_q)
        return result
