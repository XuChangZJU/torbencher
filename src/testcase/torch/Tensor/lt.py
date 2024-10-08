import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.lt)
class TorchTensorLtTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_torch_Tensor_lt_correctness(self):
        """
        Test the correctness of torch.Tensor.lt with small scale random parameters.
        """
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        input_tensor = torch.randn(input_size)  # Random tensor
        other_tensor = torch.randn(input_size)  # Random tensor with the same size as input_tensor
        result = input_tensor.lt(other_tensor)
        return result
