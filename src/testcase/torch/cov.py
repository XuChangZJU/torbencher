import torch
import random
    # Compute covariance matrix with frequency weights and importance weights


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cov)
class TorchCovTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cov_correctness(self):
        # Random dimension for the tensor (at least 2x2 matrix for meaningful covariance computation)
        num_vars = random.randint(2, 4)  # Number of variables
        num_obs = random.randint(2, 5)  # Number of observations per variable
    
        # Generate random input tensor of shape (num_vars, num_obs)
        input_tensor = torch.randn(num_vars, num_obs)
        
        # Generate random fweights (frequencies) tensor
        fweights = torch.randint(1, 10, (num_obs,))
        
        # Generate random aweights (weights) tensor
        aweights = torch.rand(num_obs)
    
        # Compute covariance matrix with default correction (Bessel's correction)
        result_default = torch.cov(input_tensor)
        
        # Compute covariance matrix with correction set to 0
        result_no_correction = torch.cov(input_tensor, correction=0)
        
        # Compute covariance matrix with frequency weights and importance weights
        result_weighted = torch.cov(input_tensor, fweights=fweights, aweights=aweights)
        
        return result_default, result_no_correction, result_weighted
    
    
    
    
    