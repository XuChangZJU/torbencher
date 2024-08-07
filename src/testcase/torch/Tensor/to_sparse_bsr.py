import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_sparse_bsr)
class TorchTensorToUsparseUbsrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_to_sparse_bsr_correctness(self):
        # Randomly generate dimensions for the tensor
        dim = random.randint(2, 4)  # Minimum 2 dimensions for sparse tensor
        num_of_elements_each_dim = random.randint(1, 5)  # Random number of elements each dimension
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Generate a random dense tensor
        dense_tensor = torch.randn(input_size)

        # Randomly generate block size ensuring it divides the first two dimensions
        blocksize = (random.randint(1, num_of_elements_each_dim), random.randint(1, num_of_elements_each_dim))

        # Ensure blocksize divides the first two dimensions
        while input_size[0] % blocksize[0] != 0 or input_size[1] % blocksize[1] != 0:
            blocksize = (random.randint(1, num_of_elements_each_dim), random.randint(1, num_of_elements_each_dim))

        # Randomly generate dense_dim ensuring it is between 0 and dim - 2
        dense_dim = random.randint(0, dim - 2)

        # Convert the dense tensor to sparse BSR format
        sparse_bsr_tensor = dense_tensor.to_sparse_bsr(blocksize, dense_dim)

        return sparse_bsr_tensor
