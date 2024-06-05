
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_allocated)
class TorchCudaMemoryAllocatedTestCase(TorBencherTestCaseBase):
    def test_memory_allocated_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_allocated(device)
        return result

    def test_memory_allocated_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_allocated(device)
        return result

