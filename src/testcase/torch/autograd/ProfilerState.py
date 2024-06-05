
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.autograd.ProfilerState)
class TorchAutogradProfilerStateTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_profilerstate_correctness(self):
        profiler_state = torch.autograd.ProfilerState(random.choice(["CPU", "CUDA", "CUSTOM", "ALL"]))
        result = profiler_state
        return result

    @test_api_version.larger_than("1.1.3")
    def test_profilerstate_large_scale(self):
        profiler_state = torch.autograd.ProfilerState(random.choice(["CPU", "CUDA", "CUSTOM", "ALL"]))
        result = profiler_state
        return result


