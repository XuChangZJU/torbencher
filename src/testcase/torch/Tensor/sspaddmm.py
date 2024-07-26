import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sspaddmm)
class TorchTensorSspaddmmTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sspaddmm_correctness(self):
        # Random dimensions for the sparse matrix
        sparse_size = [random.randint(1, 5) for _ in range(2)]  # Sparse matrix should be 2D

        # Random dimensions for the dense matrices
        dense_num_elements_each_dim1 = random.randint(1, 5)
        dense_size1 = [sparse_size[0], dense_num_elements_each_dim1]  # Ensure compatibility for multiplication

        dense_size2 = [dense_num_elements_each_dim1, sparse_size[1]]  # Ensure compatibility for multiplication

        # Create random sparse matrix
        sparse_indices = torch.randint(0, min(sparse_size), (2, 1))
        sparse_values = torch.randn(1)
        sparse_matrix = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size)

        # Create random dense matrices
        dense_matrix1 = torch.randn(dense_size1)
        dense_matrix2 = torch.randn(dense_size2)

        # Random alpha and beta values
        alpha = random.uniform(0.1, 10.0)
        beta = random.uniform(0.1, 10.0)

        print(dense_matrix1.shape)
        print(dense_matrix2.shape)

        # Perform the sspaddmm operation
        result = sparse_matrix.sspaddmm(dense_matrix1, dense_matrix2, beta=beta, alpha=alpha)
        return result
