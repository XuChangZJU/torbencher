import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.ifft2)
class TorchFftIfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_ifft2_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(2, 4)  # Dimension of the tensor should be at least 2 for ifft2
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random complex tensor
        input_tensor = torch.randn(input_size, dtype=torch.complex64)

        # Perform ifft2
        result = torch.fft.ifft2(input_tensor)
        return result
