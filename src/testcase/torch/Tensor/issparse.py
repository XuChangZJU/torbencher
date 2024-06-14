import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.Tensor.issparse)
class TorchTensorIssparseTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_sparse_correctness(self):
    # Randomly generate input size for the tensor
    dim = random.randint(1, 4)
    num_of_elements_each_dim = random.randint(1, 5)
    input_size = [num_of_elements_each_dim for i in range(dim)]

    # Create a dense tensor
    dense_tensor = torch.randn(input_size)
    
    # Create a sparse tensor
    i = torch.LongTensor([[0, 1, 1],
                       [2, 0, 2]])
    v = torch.FloatTensor([3, 4, 5])
    sparse_tensor = torch.sparse_coo_tensor(i, v, [2, 3])

    # Check if the dense tensor is sparse (should be False)
    result_dense = dense_tensor.is_sparse
    
    # Check if the sparse tensor is sparse (should be True)
    result_sparse = sparse_tensor.is_sparse

    return result_dense, result_sparse
