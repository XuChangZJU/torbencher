import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.wishart.Wishart)
class TorchDistributionsWishartWishartTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_distributions_wishart_Wishart_correctness(self):
        # generate random valid parameters
        dim = random.randint(2, 5)  # dim should be larger than 1
        df = random.uniform(float(dim), float(dim) + 10.0)  # df should be larger than dim - 1
        covariance_matrix = torch.rand(dim, dim)
        covariance_matrix = covariance_matrix @ covariance_matrix.t() + torch.eye(
            dim) * 1e-6  # make covariance_matrix positive-definite

        # instantiate Wishart distribution
        wishart_dist = torch.distributions.wishart.Wishart(df, covariance_matrix=covariance_matrix)

        # sample from the distribution
        sample = wishart_dist.sample()

        # return the sample
        return sample
