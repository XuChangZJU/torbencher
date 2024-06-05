
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.max_memory_reserved)
class TorchCudaMaxMemoryReservedTestCase(TorBencherTestCaseBase):
    def test_max_memory_reserved_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.max_memory_reserved(device)
        return result

    def test_max_memory_reserved_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.max_memory_reserved(device)
        return result

