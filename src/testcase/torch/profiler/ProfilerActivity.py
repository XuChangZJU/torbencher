
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.ProfilerActivity)
class TorchProfilerActivityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_ProfilerActivity_pybind11_type(self):
        result = torch.profiler.ProfilerActivity.pybind11_type
        return result

    @test_api_version.larger_than("1.12.0")
    def test_ProfilerActivity_name(self):
        result = torch.profiler.ProfilerActivity.name
        return result

