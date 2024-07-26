import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api



@test_api(torch.profiler.ProfilerAction)
class TorchProfilerProfileractionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_profiler_action_correctness(self):
        # No random parameters needed for ProfilerAction
        profiler_action = torch.profiler.ProfilerAction.NONE # The possible returned values are: NONE, WARMUP, RECORD, RECORD_AND_SAVE
        return profiler_action
    