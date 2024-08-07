import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler_util.Interval)
class TorchAutogradProfilerUutilIntervalTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_Interval_correctness(self):
        # Generate random start and end times
        start = random.uniform(0, 10)  # Random start time between 0 and 10
        end = random.uniform(start, start + 10)  # Random end time after start

        # Create an Interval object
        interval = torch.autograd.profiler_util.Interval(start, end)

        return interval
