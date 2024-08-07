import unittest

import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sparse_csc_tensor)
class TorchSparseUcscUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    @unittest.skip
    def test_sparse_csc_tensor_correctness(self):
        # Random number of columns and rows for the sparse matrix
        num_cols = random.randint(2, 5)
        num_rows = random.randint(2, 5)

        # Random number of non-zero elements
        num_nonzeros = random.randint(1, num_cols * num_rows)

        # Random ccol_indices satisfying the CSC format requirements
        ccol_indices = [0]
        for _ in range(num_cols):
            next_index = random.randint(ccol_indices[-1], num_nonzeros)
            ccol_indices.append(next_index)
        ccol_indices = torch.tensor(ccol_indices, dtype=torch.int64)

        # Random row_indices and values
        # row_indices = torch.tensor(random.sample(range(num_rows) * (num_nonzeros // num_rows + 1), num_nonzeros), dtype=torch.int64)
        row_indices = torch.tensor(
            random.sample(list(range(num_rows)) * (num_nonzeros // num_rows + 1), num_nonzeros),
            dtype=torch.int64
        )
        values = torch.randn(num_nonzeros)  # Random values size: [num_nonzeros]

        # Random size of the sparse tensor
        size = (num_rows, num_cols)

        result = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size)
        return result
