import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.irfft)
class TorchFftIrfftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_irfft_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate a random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Append a dimension for the last dimension
        input_size.append(random.randint(2, 10) * 2 + 1)
    
        # Generate a random complex tensor
        input_tensor = torch.randn(input_size, dtype=torch.cfloat)
    
        # Perform irfft operation
        result = torch.fft.irfft(input_tensor)
    
        return result
    
    
    
    