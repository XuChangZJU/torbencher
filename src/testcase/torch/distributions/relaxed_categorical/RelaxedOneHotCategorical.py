import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.relaxed_categorical.RelaxedOneHotCategorical)
class TorchDistributionsRelaxedUcategoricalRelaxedonehotcategoricalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_RelaxedOneHotCategorical_correctness(self):
        """
        Test the correctness of RelaxedOneHotCategorical with small scale random parameters.
        """
        batch_size = random.randint(1, 10)
        num_events = random.randint(2, 10)  # Ensure at least two events
        temperature = random.uniform(0.1, 10.0)
        probs = torch.randn(batch_size, num_events).softmax(dim=-1)  # Ensure probabilities sum to 1

        distribution = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(temperature, probs)
        sample = distribution.sample()

        return sample
