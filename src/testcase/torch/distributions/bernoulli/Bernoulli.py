import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.bernoulli.Bernoulli)
class TorchDistributionsBernoulliBernoulliTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_bernoulli_correctness(self):
        """
        Test the correctness of the Bernoulli distribution with small scale random parameters.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random probabilities between 0 and 1
        probs = torch.rand(input_size) 
        
        # Create a Bernoulli distribution
        bernoulli_distribution = torch.distributions.bernoulli.Bernoulli(probs)
    
        # Sample from the distribution
        samples = bernoulli_distribution.sample()
        return samples
    