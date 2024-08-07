import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.enable_mem_efficient_sdp)
class TorchBackendsCudaCufftUplanUcacheTorchBackendsCudaEnableUmemUefficientUsdpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")

        # Randomly generate the number of plans to store in the cache
        max_size = random.randint(1, 10)
        torch.backends.cuda.cufft_plan_cache.max_size = max_size

        # Generate random tensor sizes for FFT operations
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create random tensors for FFT operations
        tensor = torch.randn(input_size, dtype=torch.complex64, device='cuda')

        # Perform FFT operation to populate the plan cache
        fft_result = torch.fft.fft(tensor)

        # Check the size of the plan cache
        cache_size = torch.backends.cuda.cufft_plan_cache.size
        return cache_size

    def test_enable_mem_efficient_sdp(self):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")

        # Enable memory efficient SDP
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # Generate random tensor sizes for SDP operations
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create random tensors for SDP operations
        tensor1 = torch.randn(input_size, device='cuda')
        tensor2 = torch.randn(input_size, device='cuda')

        # Perform SDP operation
        sdp_result = torch.nn.functional.scaled_dot_product_attention(tensor1, tensor2, tensor2)

        # Check if memory efficient SDP is enabled
        mem_efficient_sdp_enabled = torch.backends.cuda.is_mem_efficient_sdp_enabled()
        return mem_efficient_sdp_enabled
