import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.cudnn_sdp_enabled)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaCudnnsdpenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch is compiled with CUDA support.")

        # Randomly generate the size of the cache
        max_size = random.randint(1, 10)
        torch.backends.cuda.cufft_plan_cache.max_size = max_size

        # Generate random tensor sizes
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create random tensors on CUDA
        tensor1 = torch.randn(input_size, device='cuda')
        tensor2 = torch.randn(input_size, device='cuda')

        # Perform FFT operations to populate the cache
        fft_result1 = torch.fft.fft(tensor1)
        fft_result2 = torch.fft.fft(tensor2)

        # Check the cache size
        cache_size = len(torch.backends.cuda.cufft_plan_cache)

        return cache_size, max_size

    def test_cudnn_sdp_enabled(self):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please ensure that your PyTorch is compiled with CUDA support.")

        # Check the current state of cudnn_sdp_enabled
        initial_state = torch.backends.cuda.cudnn_sdp_enabled()

        # Toggle the state
        torch.backends.cuda.cudnn_sdp_enabled(not initial_state)
        toggled_state = torch.backends.cuda.cudnn_sdp_enabled()

        # Reset to initial state
        torch.backends.cuda.cudnn_sdp_enabled(initial_state)
        reset_state = torch.backends.cuda.cudnn_sdp_enabled()

        return initial_state, toggled_state, reset_state
