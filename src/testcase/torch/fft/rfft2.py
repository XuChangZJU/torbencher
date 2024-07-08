import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.fft.rfft2)
class TorchFftRfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rfft2_correctness(self):
        # Generate random dimension for the input tensor
        dim = random.randint(2, 4)
        # Generate random number of elements for each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Create input size list
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random input tensor
        input_tensor = torch.randn(input_size)
        # Calculate rfft2
        result = torch.fft.rfft2(input_tensor)
        return result
    