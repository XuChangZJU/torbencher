import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.Tensor.to_sparse_bsc)
class TorchTensorToUsparseUbscTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_to_sparse_bsc_correctness(self):
        dense = torch.randn(10, 10)
        sparse = dense.to_sparse_csr()
        sparse_bsc = sparse.to_sparse_bsc((5, 5))
        return sparse_bsc