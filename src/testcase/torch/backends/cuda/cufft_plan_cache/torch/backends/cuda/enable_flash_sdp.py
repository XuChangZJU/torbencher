import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.enable_flash_sdp)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaEnableflashsdpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Randomly generate the number of plans to cache
        max_size = random.randint(1, 100)
        
        # Set the max size of the plan cache
        torch.backends.cuda.cufft_plan_cache.max_size = max_size
        
        # Retrieve the current max size to verify correctness
        current_max_size = torch.backends.cuda.cufft_plan_cache.max_size
        
        return current_max_size
    def test_enable_flash_sdp_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Enable flash SDP
        torch.backends.cuda.enable_flash_sdp(True)
        
        # Check if flash SDP is enabled
        is_enabled = torch.backends.cuda.is_flash_sdp_enabled()
        
        return is_enabled
    