import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.sdp_kernel)
class TorchBackendsCudaCufftUplanUcacheTorchBackendsCudaSdpUkernelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Please ensure that your PyTorch installation is compiled with CUDA support.")

        # Randomly generate the number of plans to cache
        max_size = random.randint(1, 10)

        # Set the max size of the plan cache
        torch.backends.cuda.cufft_plan_cache.max_size = max_size

        # Generate a random tensor size for FFT operation
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Create a random tensor on CUDA
        tensor = torch.randn(input_size, dtype=torch.complex64, device='cuda')

        # Perform FFT operation to populate the plan cache
        fft_result = torch.fft.fft(tensor)

        # Check the current size of the plan cache
        current_size = torch.backends.cuda.cufft_plan_cache.size

        return max_size, current_size, fft_result
