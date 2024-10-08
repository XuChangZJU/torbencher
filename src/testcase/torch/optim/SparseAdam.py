import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.optim.SparseAdam)
class TorchOptimSparseadamTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sparse_adam_correctness(self):
        # Random dimension for the tensors
        dim = random.randint(1, 4)
        # Random number of elements each dimension
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for _ in range(dim)]

        # Calculate the number of non-zero elements
        nnz = num_of_elements_each_dim

        # Generate random indices for the sparse tensor
        indices = torch.randint(0, num_of_elements_each_dim, (dim, nnz))
        values = torch.randn(nnz)
        sparse_grad = torch.sparse_coo_tensor(indices, values, input_size)

        # Random dense tensor for parameters
        params = torch.randn(input_size, requires_grad=True)

        # Initialize SparseAdam optimizer
        optimizer = torch.optim.SparseAdam([params])

        # Perform a single optimization step
        params.grad = sparse_grad
        optimizer.step()

        return params
