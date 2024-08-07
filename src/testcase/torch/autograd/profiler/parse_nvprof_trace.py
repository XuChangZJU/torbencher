import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.parse_nvprof_trace)
class TorchAutogradProfilerParseUnvprofUtraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_parse_nvprof_trace_correctness(self):
        # Generate a random tensor to simulate a trace file content
        trace_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))

        # Convert tensor to a string to simulate a trace file content
        trace_str = trace_tensor.numpy().tobytes()

        # Parse the trace string using the profiler
        result = torch.autograd.profiler.parse_nvprof_trace(trace_str.decode('latin1'))

        return result
