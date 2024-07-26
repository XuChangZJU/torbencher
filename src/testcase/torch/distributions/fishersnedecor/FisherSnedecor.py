import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.distributions.fishersnedecor.FisherSnedecor)
class TorchDistributionsFishersnedecorFishersnedecorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fisher_snedecor_correctness(self):
        # Random degrees of freedom parameters for the Fisher-Snedecor distribution
        df1 = torch.tensor([random.uniform(0.1, 10.0)])  # df1 should be a positive float
        df2 = torch.tensor([random.uniform(0.1, 10.0)])  # df2 should be a positive float
    
        # Create the Fisher-Snedecor distribution
        fisher_snedecor_dist = torch.distributions.fishersnedecor.FisherSnedecor(df1, df2)
        
        # Sample from the distribution
        sample = fisher_snedecor_dist.sample()
        
        return sample
    