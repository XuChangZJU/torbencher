
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.tensorboard_trace_handler)
class TorchTensorboardTraceHandlerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_tensorboard_trace_handler_correctness(self):
        result = torch.profiler.tensorboard_trace_handler
        return result

    @test_api_version.larger_than("1.12.0")
    def test_tensorboard_trace_handler_large_scale(self):
        result = torch.profiler.tensorboard_trace_handler
        return result



