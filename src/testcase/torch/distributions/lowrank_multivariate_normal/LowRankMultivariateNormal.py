import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal)
class TorchDistributionsLowrankUmultivariateUnormalLowrankmultivariatenormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_LowRankMultivariateNormal_correctness(self):
        # Randomly generate batch_shape, event_shape and rank
        batch_shape_dim = random.randint(1, 3)
        event_shape_dim = random.randint(1, 3)
        rank = random.randint(1, 3)
        batch_shape_size = [random.randint(1, 5) for _ in range(batch_shape_dim)]
        event_shape_size = [random.randint(1, 5) for _ in range(event_shape_dim)]

        # Generate random parameters for LowRankMultivariateNormal
        loc = torch.randn(batch_shape_size + event_shape_size)
        cov_factor = torch.randn(batch_shape_size + event_shape_size + [rank])
        cov_diag = torch.rand(batch_shape_size + event_shape_size) + 1e-5  # Ensure cov_diag is positive

        # Create LowRankMultivariateNormal distribution
        m = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc, cov_factor, cov_diag)

        # Sample from the distribution
        sample = m.sample()
        return sample
