import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.binomial.Binomial)
class TorchDistributionsBinomialBinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_binomial_Binomial_correctness(self):
        # Generate random parameters for total_count
        total_count = random.randint(1, 100)  # Random integer between 1 and 100
    
        # Generate random parameters for probs
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        probs = torch.rand(input_size)  # Random tensor with values between 0 and 1
    
        # Create a Binomial distribution
        binomial_distribution = torch.distributions.binomial.Binomial(total_count, probs)
    
        # Sample from the distribution
        sample = binomial_distribution.sample()
        return sample
    