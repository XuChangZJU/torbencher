import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributions.multinomial.Multinomial)
class TorchDistributionsMultinomialMultinomialTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_multinomial_Multinomial_correctness(self):
        # Generate random parameters for Multinomial distribution
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        total_count = random.randint(1, 100)  # Number of trials
        probs = torch.randn(input_size).abs()  # Event probabilities (non-negative)
        probs /= probs.sum(dim=-1, keepdim=True)  # Normalize probabilities to sum to 1
    
        # Create a Multinomial distribution
        multinomial_distribution = torch.distributions.multinomial.Multinomial(total_count, probs)
    
        # Sample from the distribution
        sample = multinomial_distribution.sample()
        return sample
    
    
    
    