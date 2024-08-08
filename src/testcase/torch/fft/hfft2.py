import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fft.hfft2)
class TorchFftHfft2TestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_hfft2_correctness(self):
        """
        Test the correctness of torch.fft.hfft2 with small scale random parameters.
        """
        dim = random.randint(2, 4)  # Dimension of the tensor, at least 2 for hfft2
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]
        last_dim_size = input_size[-1]

        # Ensure last dimension size is (2^n + 1) to support default 's' in hfft2
        input_size[-1] = 2 ** (random.randint(1, 5)) + 1

        input_tensor = torch.randn(input_size)
        # Make the input tensor Hermitian-symmetric in the last two dimensions
        input_tensor = input_tensor + torch.conj(torch.flip(input_tensor, dims=[-1, -2]))

        # Calculate hfft2
        result = torch.fft.hfft2(input_tensor)
        return result
