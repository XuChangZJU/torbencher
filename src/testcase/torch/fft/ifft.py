import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.ifft)
class TorchFftIfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ifft_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = 2**(random.randint(1,5)) # The signal length should be a power of 2
        input_size=[num_of_elements_each_dim for i in range(dim)] 
    
        # Generate random complex tensor
        input_tensor = torch.randn(input_size) + 1j * torch.randn(input_size)
        
        # Apply ifft
        result = torch.fft.ifft(input_tensor)
        
        return result
    