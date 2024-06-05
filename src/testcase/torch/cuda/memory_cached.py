
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_cached)
class TorchCudaMemoryCachedTestCase(TorBencherTestCaseBase):
    def test_memory_cached_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_cached(device)
        return result

    def test_memory_cached_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_cached(device)
        return result

