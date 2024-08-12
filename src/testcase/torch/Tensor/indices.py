import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.indices)
class TorchTensorIndicesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_indices_correctness(self):
        # Randomly generate dimensions for the sparse tensor
        dim = random.randint(1, 4)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random indices and values for the sparse tensor
        indices = torch.randint(0, num_of_elements_each_dim, (dim, num_of_elements_each_dim))
        values = torch.randn(num_of_elements_each_dim)

        # Create a sparse COO tensor
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=input_size).coalesce()

        # Get the indices tensor
        result = sparse_tensor.indices()
        return result
