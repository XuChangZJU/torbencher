import unittest

import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sparse_bsr_tensor)
class TorchSparseUbsrUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    @unittest.skip
    def test_sparse_bsr_tensor_correctness(self):
        # Generate random parameters for sparse_bsr_tensor
        batch_size = random.randint(1, 3)
        nrowblocks = random.randint(1, 3)
        ncolblocks = random.randint(1, 3)
        block_size = (
        random.randint(1, 3), random.randint(1, 3))  # blocksize[0] and blocksize[1] should be greater than 0
        dense_dims = (
        random.randint(1, 3), random.randint(1, 3))  # densesize[0] and densesize[1] should be greater than 0

        # Generate crow_indices
        crow_indices = [0]
        for i in range(batch_size):
            for j in range(nrowblocks):
                crow_indices.append(crow_indices[-1] + random.randint(1,
                                                                      3))  # Each element should be greater than or equal to the previous element
        crow_indices = torch.tensor(crow_indices, dtype=torch.int64)

        # Generate col_indices
        total_blocks = crow_indices[-1].item()
        col_indices = torch.randint(0, ncolblocks, (total_blocks,), dtype=torch.int64)

        # Generate values
        values = torch.randn((total_blocks, block_size[0], block_size[1], dense_dims[0], dense_dims[1]))

        # Calculate size
        size = (batch_size, nrowblocks * block_size[0], ncolblocks * block_size[1], dense_dims[0], dense_dims[1])

        # Create sparse_bsr_tensor
        sparse_tensor = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size)
        return sparse_tensor
