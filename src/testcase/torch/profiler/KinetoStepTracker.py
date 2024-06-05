
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.KinetoStepTracker)
class TorchKinetoStepTrackerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_KinetoStepTracker_type(self):
        result = torch.profiler.KinetoStepTracker.type
        return result

    @test_api_version.larger_than("1.12.0")
    def test_KinetoStepTracker_current_step(self):
        result = torch.profiler.KinetoStepTracker.current_step
        return result

    @test_api_version.larger_than("1.12.0")
    def test_KinetoStepTracker_erase_step_count(self):
        result = torch.profiler.KinetoStepTracker.erase_step_count
        return result

    @test_api_version.larger_than("1.12.0")
    def test_KinetoStepTracker_increment_step(self):
        result = torch.profiler.KinetoStepTracker.increment_step()
        return result

    @test_api_version.larger_than("1.12.0")
    def test_KinetoStepTracker_init_step_count(self):
        result = torch.profiler.KinetoStepTracker.init_step_count
        return result

