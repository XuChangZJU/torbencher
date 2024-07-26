import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.multivariate_normal.MultivariateNormal)
class TorchDistributionsMultivariatenormalMultivariatenormalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_multivariate_normal_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the mean vector and covariance matrix
        mean_vector = torch.randn(dim)  # Random mean vector
        covariance_matrix = torch.randn(dim, dim)  # Random covariance matrix
        covariance_matrix = torch.mm(covariance_matrix, covariance_matrix.t())  # Make it positive-definite

        distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean_vector, covariance_matrix)
        sample = distribution.sample()  # Sample from the distribution
        return sample
