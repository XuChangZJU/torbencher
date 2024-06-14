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
    sparse_dim = random.randint(1, 4)
    sparse_num_elements_each_dim = random.randint(1, 5)
    sparse_size = [sparse_num_elements_each_dim for _ in range(sparse_dim)]
    
    # Random dimensions for the dense matrices
    dense_dim1 = random.randint(1, 4)
    dense_num_elements_each_dim1 = random.randint(1, 5)
    dense_size1 = [dense_num_elements_each_dim1 for _ in range(dense_dim1)]
    
    dense_dim2 = random.randint(1, 4)
    dense_num_elements_each_dim2 = random.randint(1, 5)
    dense_size2 = [dense_num_elements_each_dim2 for _ in range(dense_dim2)]
    
    # Ensure the inner dimensions match for matrix multiplication
    dense_size2[0] = dense_size1[-1]
    
    # Create random sparse matrix
    sparse_indices = torch.randint(0, sparse_num_elements_each_dim, (2, sparse_num_elements_each_dim))
    sparse_values = torch.randn(sparse_num_elements_each_dim)
    sparse_matrix = torch.sparse_coo_tensor(sparse_indices, sparse_values, sparse_size)
    
    # Create random dense matrices
    dense_matrix1 = torch.randn(dense_size1)
    dense_matrix2 = torch.randn(dense_size2)
    
    # Random alpha and beta values
    alpha = random.uniform(0.1, 10.0)
    beta = random.uniform(0.1, 10.0)
    
    # Perform the sspaddmm operation
    result = torch.sspaddmm(sparse_matrix, dense_matrix1, dense_matrix2, beta=beta, alpha=alpha)
    return result
