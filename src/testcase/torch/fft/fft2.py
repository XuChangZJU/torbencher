import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.fft2)
class TorchFftFft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_fft2_correctness(self):
        # Define the dimension of the input tensor
        dim = random.randint(2, 4)
        # Define the size of each dimension, ensuring the last two dimensions are valid for FFT
        input_size = [random.randint(1, 5) for _ in range(dim - 2)] + [2 ** random.randint(1, 5),
                                                                       2 ** random.randint(1, 5)]
        # Generate a random complex tensor
        input_tensor = torch.randn(input_size, dtype=torch.complex64)
        # Perform 2D FFT
        result = torch.fft.fft2(input_tensor)
        return result
