import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.math_sdp_enabled)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaMathsdpenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        # Randomly generate the size of the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random tensor
        tensor = torch.randn(input_size)

        # Check the current state of math_sdp_enabled
        math_sdp_enabled = torch.backends.cuda.math_sdp_enabled()

        # Perform some operation that would be affected by math_sdp_enabled
        # For demonstration, let's assume we are performing a simple FFT operation
        if math_sdp_enabled:
            result = torch.fft.fft(tensor)
        else:
            result = torch.fft.ifft(tensor)

        return result
