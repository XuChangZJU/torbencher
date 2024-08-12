import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sparse_csr_tensor)
class TorchSparseUcsrUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sparse_csr_tensor_correctness(self):
        # Random dimension sizes for the sparse matrix
        nrows = random.randint(2, 5)
        ncols = random.randint(2, 5)

        # Random number of non-zeros per row
        nz_per_row = random.randint(1, ncols)

        # Crow indices: Starting indices of each row plus final number of non-zero elements
        crow_indices = [0]
        for _ in range(nrows):
            crow_indices.append(crow_indices[-1] + nz_per_row)

        crow_indices = torch.tensor(crow_indices, dtype=torch.int64)

        # Column indices: Random column indices for each non-zero element
        col_indices = []
        for _ in range(nrows * nz_per_row):
            col_indices.append(random.randint(0, ncols - 1))

        col_indices = torch.tensor(col_indices, dtype=torch.int64)

        # Values: Random values for each non-zero element
        values = torch.randn(nrows * nz_per_row)

        # Create sparse CSR tensor
        result = torch.sparse_csr_tensor(crow_indices, col_indices, values)

        return result
