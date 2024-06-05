
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.schedule)
class TorchScheduleTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_schedule_correctness(self):
        result = torch.profiler.schedule
        return result

    @test_api_version.larger_than("1.12.0")
    def test_schedule_large_scale(self):
        result = torch.profiler.schedule
        return result

