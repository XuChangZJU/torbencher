import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.index_add_)
class TorchTensorIndexUaddUTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_index_add_correctness(self):
        dim = random.randint(0, 2)  # Random dimension for the operation (0, 1, or 2)
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(3)]  # 3D tensor size

        # Create the self tensor
        self_tensor = torch.randn(input_size)

        # Create the source tensor
        source_size = input_size.copy()
        source_size[dim] = random.randint(1,
                                          5)  # Ensure source tensor has the same size as index length in the specified dimension
        source_tensor = torch.randn(source_size)

        # Create the index tensor
        index_length = source_size[dim]
        index_tensor = torch.randint(0, input_size[dim], (index_length,), dtype=torch.int64)

        # Perform the index_add_ operation
        result = self_tensor.index_add_(dim, index_tensor, source_tensor)
        return result
