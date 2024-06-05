
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.LongStorage)
class TorchCudaLongStorageTestCase(TorBencherTestCaseBase):
    def test_long_storage_correctness(self):
        size = random.randint(1, 10)
        data = [random.randint(-100, 100) for _ in range(size)]
        result = torch.cuda.LongStorage.from_list(data)
        return result

    def test_long_storage_large_scale(self):
        size = random.randint(1000, 10000)
        data = [random.randint(-100, 100) for _ in range(size)]
        result = torch.cuda.LongStorage.from_list(data)
        return result

