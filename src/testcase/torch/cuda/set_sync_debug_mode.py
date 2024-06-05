
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.set_sync_debug_mode)
class TorchCudaSetSyncDebugModeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.5.0")
    def test_set_sync_debug_mode_correctness(self):
        mode = random.choice([True, False])
        result = torch.cuda.set_sync_debug_mode(mode)
        return result

    @test_api_version.larger_than("1.5.0")
    def test_set_sync_debug_mode_large_scale(self):
        mode = random.choice([True, False])
        result = torch.cuda.set_sync_debug_mode(mode)
        return result

