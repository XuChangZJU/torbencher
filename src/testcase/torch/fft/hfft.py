import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.hfft)
class TorchFftHfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_hfft_correctness(self):
        # Random dimension for the tensor
        dim = random.randint(1, 4)
        
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        
        # Input size for the tensor
        input_size = [num_of_elements_each_dim for _ in range(dim)]
        
        # Generate a random tensor with Hermitian symmetry
        input_tensor = torch.randn(input_size, dtype=torch.complex64)
        
        # Randomly choose the output signal length
        n = random.randint(1, 10)
        
        # Randomly choose the dimension along which to take the FFT
        fft_dim = random.randint(0, dim - 1)
        
        # Compute the Hermitian FFT
        result = torch.fft.hfft(input_tensor, n=n, dim=fft_dim)
        
        return result
    