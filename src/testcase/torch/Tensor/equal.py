import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.equal)
class TorchTensorEqualTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tensor_equal_correctness(self):
        dim = random.randint(1, 4)  # Random dimension for the tensors
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]  # Generate input size list

        tensor1 = torch.randn(input_size)  # Generate first random tensor
        tensor2 = torch.randn(input_size)  # Generate second random tensor

        # Check if the two tensors are equal
        result = tensor1.equal(tensor2)
        return result
