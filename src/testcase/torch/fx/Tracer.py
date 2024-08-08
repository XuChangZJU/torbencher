import math

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.fx.Tracer)
class TorchFxTracerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_tracer_correctness(self):
        # Randomly generate parameters for Tracer
        autowrap_modules = (math,)
        autowrap_functions = ()

        # Create a Tracer instance
        tracer = torch.fx.Tracer(autowrap_modules, autowrap_functions)
        return tracer
