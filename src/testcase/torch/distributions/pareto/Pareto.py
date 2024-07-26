import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.pareto.Pareto)
class TorchDistributionsParetoParetoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_pareto_Pareto_correctness_with_small_random_scale(self):
        # Generate random dimension for the tensors
        dim = random.randint(1, 4)
        # Generate random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random scale, which should be greater than 0
        scale = torch.rand(input_size) + 1e-5
        # Generate random alpha, which should be greater than 0
        alpha = torch.rand(input_size) + 1e-5
        # Generate Pareto distribution
        pareto_distribution = torch.distributions.pareto.Pareto(scale, alpha)
        # Sample from the distribution
        sample = pareto_distribution.sample()
        # Return the sample
        return sample
