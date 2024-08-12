import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.signbit)
class TorchTensorSignbitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_signbit_correctness(self):
        """
        Test the correctness of the signbit function on a small random tensor.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        tensor = torch.randn(input_size)  # Generate random tensor with elements from standard normal distribution
        result = tensor.signbit()
        return result
