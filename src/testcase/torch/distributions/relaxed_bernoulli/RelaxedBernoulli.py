import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.distributions.relaxed_bernoulli.RelaxedBernoulli)
class TorchDistributionsRelaxedbernoulliRelaxedbernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_relaxed_bernoulli_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Random temperature
        temperature = torch.randn(input_size)
        # Random probs
        probs = torch.rand(input_size)  # probs should be in range [0, 1]
        # Create RelaxedBernoulli distribution
        relaxed_bernoulli_distribution = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, probs)
        # Sample from the distribution
        result = relaxed_bernoulli_distribution.sample()
        return result
