import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_sparse)
class TorchTensorToUsparseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_to_sparse_correctness(self):
        # Randomly generate tensor dimension and size
        dim = random.randint(2, 4)  # Dimension should be at least 2 for sparse tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random dense tensor
        dense_tensor = torch.randn(input_size)

        # Randomly choose the number of sparse dimensions
        sparse_dims = random.randint(1, dim)  # sparseDims should be within the tensor's dimensions

        # Convert to sparse tensor
        sparse_tensor = dense_tensor.to_sparse(sparse_dims)
        return sparse_tensor
