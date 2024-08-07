import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_sparse_bsc)
class TorchTensorToUsparseUbscTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_to_sparse_bsc_correctness(self):
        # Randomly generate input tensor size
        dim = random.randint(2, 4)  # Dimension should be at least 2 for sparse tensors
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]

        # Generate random dense tensor
        dense_tensor = torch.randn(input_size)

        # Generate random blocksize, ensuring it evenly divides sparse dimensions
        blocksize = (random.randint(1, input_size[-2]), random.randint(1, input_size[-1]))

        # Convert to sparse BSC format
        sparse_bsc_tensor = dense_tensor.to_sparse_bsc(blocksize)

        return sparse_bsc_tensor
