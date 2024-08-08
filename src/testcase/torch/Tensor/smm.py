import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.smm)
class TorchTensorSmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_smm_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate random 2D tensor for the matrix
        matrix = torch.randn(num_of_elements_each_dim, num_of_elements_each_dim)
        # Generate random 2D tensor for the matrix to be multiplied
        mat = torch.randn(num_of_elements_each_dim, num_of_elements_each_dim)

        # Convert the first matrix to a sparse tensor
        sparse_matrix = matrix.to_sparse()

        # Perform the sparse matrix multiplication
        result = torch.sparse.mm(sparse_matrix, mat)
        return result
