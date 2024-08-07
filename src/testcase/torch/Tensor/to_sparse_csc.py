import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_sparse_csc)
class TorchTensorToUsparseUcscTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_to_sparse_csc_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(2, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        # Generate random input size
        input_size = [num_of_elements_each_dim for i in range(dim)]
        # Generate random dense tensor
        dense_tensor = torch.randn(input_size)
        # Convert to sparse CSC format
        sparse_csc_tensor = dense_tensor.to_sparse_csc()
        # Return the sparse CSC tensor
        return sparse_csc_tensor
