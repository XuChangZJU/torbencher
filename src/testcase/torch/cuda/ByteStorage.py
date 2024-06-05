
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.ByteStorage)
class TorchCudaByteStorageTestCase(TorBencherTestCaseBase):
    def test_byte_storage_correctness(self):
        size = random.randint(1, 10)
        data = [random.randint(0, 255) for _ in range(size)]
        result = torch.cuda.ByteStorage.from_list(data)
        return result

    def test_byte_storage_large_scale(self):
        size = random.randint(1000, 10000)
        data = [random.randint(0, 255) for _ in range(size)]
        result = torch.cuda.ByteStorage.from_list(data)
        return result

