
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.IntStorage)
class TorchIntStorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_intstorage_correctness(self):
        dim = random.randint(1, 10)
        result = torch.IntStorage.from_buffer(torch.randint(-1000, 1000, (dim,)).numpy())
        return result

    @test_api_version.larger_than("1.1.3")
    def test_intstorage_large_scale(self):
        dim = random.randint(1000, 10000)
        result = torch.IntStorage.from_buffer(torch.randint(-1000, 1000, (dim,)).numpy())
        return result

