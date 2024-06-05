
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.BFloat16Storage)
class TorchCudaBFloat16StorageTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.10.0")
    def test_bfloat16_storage_correctness(self):
        size = random.randint(1, 10)
        data = [random.uniform(-10.0, 10.0) for _ in range(size)]
        result = torch.cuda.BFloat16Storage.from_list(data)
        return result

    @test_api_version.larger_than("1.10.0")
    def test_bfloat16_storage_large_scale(self):
        size = random.randint(1000, 10000)
        data = [random.uniform(-10.0, 10.0) for _ in range(size)]
        result = torch.cuda.BFloat16Storage.from_list(data)
        return result

