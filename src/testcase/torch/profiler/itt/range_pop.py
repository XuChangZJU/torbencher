import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.profiler.itt.range_pop)
class TorchProfilerIttRangeUpopTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_range_pop_correctness(self):
        # No input parameters for torch.profiler.itt.range_pop
        # Manually create a nested range structure
        with torch.profiler.itt.range("outer"):
            with torch.profiler.itt.range("inner"):
                # Call range_pop to end the "inner" range
                depth_inner = torch.profiler.itt.range_pop()
            # Call range_pop again to end the "outer" range
            depth_outer = torch.profiler.itt.range_pop()
        # depth_inner should be 0 (innermost range)
        # depth_outer should be 1 
        return depth_inner, depth_outer
