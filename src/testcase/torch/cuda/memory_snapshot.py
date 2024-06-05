
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.memory_snapshot)
class TorchCudaMemorySnapshotTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_memory_snapshot_correctness(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_snapshot(device)
        return result

    @test_api_version.larger_than("1.7.0")
    def test_memory_snapshot_large_scale(self):
        device = random.randint(0, torch.cuda.device_count() - 1)
        result = torch.cuda.memory_snapshot(device)
        return result

