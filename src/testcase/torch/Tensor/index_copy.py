import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_copy)
class TorchTensorIndexUcopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_copy_correctness(self):
        dim = random.randint(0, 3)  # Random dimension to perform index_copy
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(4)]  # Generate a 4D tensor

        tensor1 = torch.randn(input_size)  # Original tensor

        index_size = random.randint(1, num_of_elements_each_dim)  # Random size for index tensor
        index = torch.randint(0, num_of_elements_each_dim, (index_size,))  # Random index tensor

        # Ensure tensor2 has the correct size for the dimension being copied
        tensor2_size = input_size.copy()
        tensor2_size[dim] = index_size
        tensor2 = torch.randn(tensor2_size)  # Tensor to copy from

        result = tensor1.index_copy(dim, index, tensor2)
        return result
