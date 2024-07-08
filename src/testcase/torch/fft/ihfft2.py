import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.ihfft2)
class TorchFftIhfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ihfft2_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(2, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input_size list for the input tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Update the last dimension to be even
        input_size[-1] = 2 * random.randint(1, 5)
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Calculate ihfft2
        result = torch.fft.ihfft2(input_tensor)
        return result
    