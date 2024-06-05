
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.sparse_coo_tensor)
class TorchSparseCooTensorTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_sparse_coo_tensor_correctness(self):
        indices = torch.randint(0, 10, (random.randint(1, 10), 2))
        values = torch.randn(random.randint(1, 10))
        result = torch.sparse_coo_tensor(indices, values, (10, 10))
        return result

    @test_api_version.larger_than("1.1.3")
    def test_sparse_coo_tensor_large_scale(self):
        indices = torch.randint(0, 1000, (random.randint(1000, 10000), 2))
        values = torch.randn(random.randint(1000, 10000))
        result = torch.sparse_coo_tensor(indices, values, (1000, 1000))
        return result

