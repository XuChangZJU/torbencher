import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sgn)
class TorchSgnTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sgn_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random complex tensor
        real_part = torch.randn(input_size)
        imag_part = torch.randn(input_size)
        complex_tensor = torch.complex(real_part, imag_part)
        result = torch.sgn(complex_tensor)
        return result
