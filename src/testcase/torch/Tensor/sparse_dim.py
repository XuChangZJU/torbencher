import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.sparse_dim)
class TorchTensorSparseUdimTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sparse_dim_correctness(self):
        # Generate a random sparse tensor
        dim = random.randint(2, 4)  # Dimension of the tensor (at least 2 for sparse)
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        sparse_dim = random.randint(1, dim)  # Number of sparse dimensions
        dense_dim = dim - sparse_dim
        indices_size = [sparse_dim, random.randint(1, num_of_elements_each_dim ** sparse_dim)]
        values_size = [random.randint(1, num_of_elements_each_dim ** sparse_dim)] + [num_of_elements_each_dim for i in
                                                                                     range(dense_dim)]
        indices = torch.randint(0, num_of_elements_each_dim, indices_size)
        values = torch.randn(values_size)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, input_size)

        result = sparse_tensor.sparse_dim()
        return result

    def test_sparse_dim_dense_tensor(self):
        # Generate a random dense tensor
        dim = random.randint(1, 4)  # Dimension of the tensor
        num_of_elements_each_dim = random.randint(1, 5)
        input_size = [num_of_elements_each_dim for i in range(dim)]
        dense_tensor = torch.randn(input_size)

        result = dense_tensor.sparse_dim()
        return result
