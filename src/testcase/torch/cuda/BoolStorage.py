
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.BoolStorage)
class TorchCudaBoolStorageTestCase(TorBencherTestCaseBase):
    def test_bool_storage_correctness(self):
        size = random.randint(1, 10)
        data = [random.choice([True, False]) for _ in range(size)]
        result = torch.cuda.BoolStorage.from_list(data)
        return result

    def test_bool_storage_large_scale(self):
        size = random.randint(1000, 10000)
        data = [random.choice([True, False]) for _ in range(size)]
        result = torch.cuda.BoolStorage.from_list(data)
        return result

