import random
import unittest

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.autograd.profiler.parse_nvprof_trace)
class TorchAutogradProfilerParseUnvprofUtraceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_parse_nvprof_trace_correctness(self):
        # Generate a random tensor to simulate a trace file content
        trace_tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))

        # Convert tensor to a string to simulate a trace file content
        trace_str = trace_tensor.numpy().tobytes()

        # Parse the trace string using the profiler
        result = torch.autograd.profiler.parse_nvprof_trace(trace_str.decode('latin1'))

        return result
