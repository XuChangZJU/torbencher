import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.complex)
class TorchComplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_complex_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        real_tensor = torch.randn(input_size)  # Real part of the complex tensor
        imag_tensor = torch.randn(input_size)  # Imaginary part of the complex tensor
        result = torch.complex(real_tensor, imag_tensor)
        return result
