import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.enable_math_sdp)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaEnablemathsdpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        # Randomly generate a size for the tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
    
        # Enable or disable the math SDP
        enable_math_sdp = random.choice([True, False])
        torch.backends.cuda.enable_math_sdp(enable_math_sdp)
    
        # Perform an operation that would be affected by the math SDP setting
        result = torch.fft.fft(tensor)
    
        return result
    