import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.fft)
class TorchFftFftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fft_correctness(self):
        # Generate random parameters for torch.fft.fft
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size, dtype=torch.complex64)  # Generate random complex tensor
    
        # Call torch.fft.fft with the generated parameters
        result = torch.fft.fft(input_tensor)
        return result
    
    
    
    