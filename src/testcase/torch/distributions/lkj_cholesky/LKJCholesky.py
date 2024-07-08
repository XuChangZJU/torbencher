import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.lkj_cholesky.LKJCholesky)
class TorchDistributionsLkjcholeskyLkjcholeskyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_lkj_cholesky_correctness(self):
        # Random dimension for the matrices
        dim = random.randint(2, 5)
        # Random concentration parameter
        concentration = random.uniform(0.1, 10.0)
        # Create an instance of LKJCholesky distribution
        lkj_cholesky = torch.distributions.lkj_cholesky.LKJCholesky(dim, concentration)
        # Sample from the distribution
        sample = lkj_cholesky.sample()
        return sample
    