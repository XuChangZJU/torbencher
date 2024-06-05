
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.DebugLevel)
class TorchDebugLevelTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_debug_level_correctness(self):
        result = torch.distributed.DebugLevel.pybind11_type()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_debug_level_large_scale(self):
        result = torch.distributed.DebugLevel.pybind11_type()
        return result

