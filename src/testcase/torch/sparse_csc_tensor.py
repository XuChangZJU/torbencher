import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.sparse_csc_tensor)
class TorchSparseUcscUtensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_sparse_csc_tensor_correctness(self):
        ccol_indices = [0, 2, 4]
        row_indices = [0, 1, 0, 1]
        values = [1, 2, 3, 4]
        return torch.sparse_csc_tensor(torch.tensor(ccol_indices, dtype=torch.int64),
                                       torch.tensor(row_indices, dtype=torch.int64),
                                       torch.tensor(values), dtype=torch.double)
