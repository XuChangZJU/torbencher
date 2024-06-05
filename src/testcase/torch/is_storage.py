
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_storage)
class TorchIsStorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_storage_correctness(self):
        storage = torch.randn(random.randint(1, 10)).storage()
        result = torch.is_storage(storage)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_is_storage_large_scale(self):
        storage = torch.randn(random.randint(1000, 10000)).storage()
        result = torch.is_storage(storage)
        return result

