import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.beta.Beta)
class TorchDistributionsBetaBetaTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_torch_distributions_beta_Beta_correctness(self):
        # Random dimension for the concentration parameters
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
    
        # Generate random concentration parameters (alpha and beta)
        concentration1 = torch.rand(input_size) + 0.1  # Ensure alpha > 0
        concentration0 = torch.rand(input_size) + 0.1  # Ensure beta > 0
    
        # Create a Beta distribution object
        beta_distribution = torch.distributions.beta.Beta(concentration1, concentration0)
    
        # Sample from the Beta distribution
        sample = beta_distribution.sample()
        return sample
    