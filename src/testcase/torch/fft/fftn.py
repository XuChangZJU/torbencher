import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.fftn)
class TorchFftFftnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_fftn_correctness(self):
        # Define the dimension of the input tensor
        dim = random.randint(1, 4)
        # Define the size of each dimension, ensuring each dimension has at least one element
        num_of_elements_each_dim = random.randint(1, 5)
        # Create the input size list for the tensor
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate a random tensor with complex data type
        input_tensor = torch.randn(input_size, dtype=torch.complex64)
        # Calculate the FFT of the input tensor
        result = torch.fft.fftn(input_tensor)
        # Return the result for testing
        return result
