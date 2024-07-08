import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.enable_cudnn_sdp)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaEnablecudnnsdpTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_cufft_plan_cache_correctness(self):
        # Randomly generate a size for the tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
    
        # Perform some operation that would use the cuFFT plan cache
        result = torch.fft.fft(tensor)
        return result
    def test_enable_cudnn_sdp_correctness(self):
        # Enable cuDNN SDP
        torch.backends.cudnn.enabled = True
        
        # Randomly generate a size for the tensor
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]
    
        # Generate a random tensor
        tensor = torch.randn(input_size)
    
        # Perform some operation that would use cuDNN SDP
        if len(tensor.shape) == 3:  # Ensure the tensor has 3 dimensions for conv2d
            tensor = tensor.unsqueeze(0)  # Add a batch dimension
        result = torch.nn.functional.conv2d(tensor, torch.randn(1, tensor.size(1), 3, 3))
        return result
    