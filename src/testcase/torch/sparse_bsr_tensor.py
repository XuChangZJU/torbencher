
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sparse_bsr_tensor)
class TorchSparseBsrTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_bsr_tensor_correctness(self):
        crow_indices = torch.randint(0, 10, (random.randint(1, 10),))
        col_indices = torch.randint(0, 10, (random.randint(1, 10),))
        values = torch.randn(random.randint(1, 10), 2, 2)
        result = torch.sparse_bsr_tensor(crow_indices, col_indices, values, (10, 10))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sparse_bsr_tensor_large_scale(self):
        crow_indices = torch.randint(0, 1000, (random.randint(1000, 10000),))
        col_indices = torch.randint(0, 1000, (random.randint(1000, 10000),))
        values = torch.randn(random.randint(1000, 10000), 2, 2)
        result = torch.sparse_bsr_tensor(crow_indices, col_indices, values, (1000, 1000))
        return result

