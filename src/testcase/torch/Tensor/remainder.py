import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.remainder)
class TorchTensorRemainderTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_remainder_correctness(self):
        """
        Test the correctness of torch.Tensor.remainder with small scale random parameters.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor data
        tensor = torch.randn(input_size)
        divisor = torch.randn(input_size)  # Divisor tensor with the same size as 'tensor'

        result = tensor.remainder(divisor)
        return result
