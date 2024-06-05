
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.ShortStorage)
class TorchShortStorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_shortstorage_correctness(self):
        dim = random.randint(1, 10)
        result = torch.ShortStorage.from_buffer(torch.randint(-32768, 32768, (dim,)).numpy())
        return result

    @test_api_version.larger_than("1.1.3")
    def test_shortstorage_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.ShortStorage.from_buffer(torch.randint(-32768, 32768, (dim,)).numpy())
        return result

