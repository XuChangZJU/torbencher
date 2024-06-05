
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.ProfilerAction)
class TorchProfilerActionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_ProfilerAction_EnumType(self):
        result = torch.profiler.ProfilerAction.EnumType
        return result

