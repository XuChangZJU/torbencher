
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.TracerWarning)
class TorchJitTracerWarningTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tracer_warning_correctness(self):
        warning = torch.jit.TracerWarning()
        result = warning.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_tracer_warning_large_scale(self):
        warning = torch.jit.TracerWarning()
        result = warning.type
        return result

