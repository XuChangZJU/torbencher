import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.irfft2)
class TorchFftIrfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_irfft2_correctness(self):
        # Randomly generate the size of each dimension
        size1 = random.randint(1, 5)
        size2 = random.randint(1, 5)
        
        # Generate a random input tensor with the specified dimensions
        input_tensor = torch.randn(size1, size2)
        
        # Perform rfft2 to get the Hermitian signal in the Fourier domain
        hermitian_signal = torch.fft.rfft2(input_tensor)
        
        # Perform irfft2 to get the inverse transform
        result = torch.fft.irfft2(hermitian_signal, s=(size1, size2))
        
        return result
    