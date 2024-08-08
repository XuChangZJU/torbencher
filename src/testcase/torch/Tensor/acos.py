import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.acos)
class TorchTensorAcosTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_acos_correctness(self):
        """
        Test the correctness of torch.Tensor.acos with small scale random parameters.
        """
        dim = random.randint(1, 4)  # Random dimension for the tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random tensor data within the range [-1, 1] to ensure valid acos operation
        input_tensor = torch.rand(input_size) * 2 - 1
        result = input_tensor.acos()
        return result
