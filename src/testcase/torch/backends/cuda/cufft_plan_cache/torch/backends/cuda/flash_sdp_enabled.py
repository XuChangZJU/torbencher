import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.flash_sdp_enabled)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaFlashsdpenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Randomly generate a size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor on GPU
        tensor = torch.randn(input_size, device='cuda')
    
        # Test the effect of torch.backends.cuda.cufft_plan_cache
        torch.backends.cuda.cufft_plan_cache.clear()
        result = torch.fft.fft(tensor)
        return result
    def test_flash_sdp_enabled_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")
        
        # Randomly generate a size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate random tensor on GPU
        tensor = torch.randn(input_size, device='cuda')
    
        # Test the effect of torch.backends.cuda.flash_sdp_enabled
        torch.backends.cuda.flash_sdp_enabled = True
        result_enabled = torch.nn.functional.softmax(tensor, dim=-1)
    
        torch.backends.cuda.flash_sdp_enabled = False
        result_disabled = torch.nn.functional.softmax(tensor, dim=-1)
    
        return result_enabled, result_disabled
    