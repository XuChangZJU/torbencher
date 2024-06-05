
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.get_debug_level)
class TorchGetDebugLevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_debug_level_correctness(self):
        result = torch.distributed.get_debug_level()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_debug_level_large_scale(self):
        result = torch.distributed.get_debug_level()
        return result

