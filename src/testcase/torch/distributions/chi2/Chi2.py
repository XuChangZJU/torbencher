import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.chi2.Chi2)
class TorchDistributionsChi2Chi2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_chi2_correctness(self):
        # Randomly generate degrees of freedom (df) for the Chi2 distribution
        df_value = random.uniform(0.1, 10.0)  # df should be a positive value
        df_tensor = torch.tensor([df_value])
        
        # Create Chi2 distribution with the generated df
        chi2_distribution = torch.distributions.chi2.Chi2(df_tensor)
        
        # Sample from the Chi2 distribution
        sample = chi2_distribution.sample()
        
        return sample
    