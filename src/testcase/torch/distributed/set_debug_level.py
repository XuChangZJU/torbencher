
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.set_debug_level)
class TorchSetDebugLevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_debug_level_correctness(self):
        debug_level = random.choice([torch.distributed.DebugLevel.OFF, torch.distributed.DebugLevel.INFO, torch.distributed.DebugLevel.DETAIL, torch.distributed.DebugLevel.ALL])
        result = torch.distributed.set_debug_level(debug_level)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_debug_level_large_scale(self):
        debug_level = random.choice([torch.distributed.DebugLevel.OFF, torch.distributed.DebugLevel.INFO, torch.distributed.DebugLevel.DETAIL, torch.distributed.DebugLevel.ALL])
        result = torch.distributed.set_debug_level(debug_level)
        return result

