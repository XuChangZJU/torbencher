import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.SDPAParams)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaSdpaparamsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Randomly generate the size of the cache
        max_size = random.randint(1, 100)
        
        # Set the max size of the plan cache
        torch.backends.cuda.cufft_plan_cache.max_size = max_size
        
        # Retrieve the current max size to verify correctness
        current_max_size = torch.backends.cuda.cufft_plan_cache.max_size
        
        # Return the current max size to check if it matches the set value
        return current_max_size
    
    def test_SDPAParams_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Randomly generate parameters for SDPAParams
        batch_size = random.randint(1, 10)
        num_heads = random.randint(1, 10)
        seq_len = random.randint(1, 10)
        head_dim = random.randint(1, 10)
        
        # Create SDPAParams object with random parameters
        sdpa_params = torch.backends.cuda.SDPAParams(batch_size, num_heads, seq_len, head_dim)
        
        # Return the SDPAParams object to verify correctness
        return sdpa_params
    
    # Run the tests
    
    
    
    