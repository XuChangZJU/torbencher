import torch
import random
from torch.distributions import Normal, kl_divergence

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.kl.register_kl)
class TorchDistributionsKlRegisterklTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_kl_correctness(self):
        # Define a custom KL divergence function for Normal distributions
        @torch.distributions.kl.register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            return torch.sum(p.log_prob(q.mean) - q.log_prob(p.mean))
    
        # Generate random parameters for Normal distributions
        mean1 = random.uniform(-10.0, 10.0)
        std1 = random.uniform(0.1, 5.0)
        mean2 = random.uniform(-10.0, 10.0)
        std2 = random.uniform(0.1, 5.0)
    
        # Create Normal distributions
        p = Normal(mean1, std1)
        q = Normal(mean2, std2)
    
        # Compute KL divergence using the registered function
        result = kl_divergence(p, q)
        return result
    