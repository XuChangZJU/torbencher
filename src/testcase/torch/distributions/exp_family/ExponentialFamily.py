import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.exp_family.ExponentialFamily)
class TorchDistributionsExpUfamilyExponentialfamilyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_exponential_family_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Randomly generate natural parameters tensor
        natural_params = torch.randn(input_size)
        # Randomly generate sufficient statistic tensor
        sufficient_stat = torch.randn(input_size)
        # Randomly generate log normalizer tensor
        log_normalizer = torch.randn(input_size)
        # Randomly generate carrier measure tensor
        carrier_measure = torch.randn(input_size)

        # Create an instance of ExponentialFamily (assuming a subclass is implemented)
        class DummyExponentialFamily(torch.distributions.exp_family.ExponentialFamily):
            def __init__(self, natural_params, sufficient_stat, log_normalizer, carrier_measure):
                self.natural_params = natural_params
                self.sufficient_stat = sufficient_stat
                self.log_normalizer = log_normalizer
                self.carrier_measure = carrier_measure

            def entropy(self):
                # Dummy entropy calculation
                return torch.sum(
                    self.natural_params * self.sufficient_stat - self.log_normalizer + self.carrier_measure)

            def kl_divergence(self, other):
                # Dummy KL divergence calculation
                return torch.sum(self.natural_params * (
                            self.sufficient_stat - other.sufficient_stat) - self.log_normalizer + other.log_normalizer)

        # Instantiate the dummy exponential family distribution
        dist = DummyExponentialFamily(natural_params, sufficient_stat, log_normalizer, carrier_measure)

        # Calculate entropy
        entropy_result = dist.entropy()
        return entropy_result
