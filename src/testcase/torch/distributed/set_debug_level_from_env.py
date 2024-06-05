
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.set_debug_level_from_env)
class TorchSetDebugLevelFromEnvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_debug_level_from_env_correctness(self):
        result = torch.distributed.set_debug_level_from_env()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_set_debug_level_from_env_large_scale(self):
        result = torch.distributed.set_debug_level_from_env()
        return result

