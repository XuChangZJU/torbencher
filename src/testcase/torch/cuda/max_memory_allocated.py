
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.max_memory_allocated)
class TorchCudaMaxMemoryAllocatedTestCase(TorBencherTestCaseBase):
    def test_max_memory_allocated_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.max_memory_allocated(device)
        return result

    def test_max_memory_allocated_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.max_memory_allocated(device)
        return result

