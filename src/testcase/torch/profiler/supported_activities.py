
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.supported_activities)
class TorchSupportedActivitiesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_supported_activities_correctness(self):
        result = torch.profiler.supported_activities
        return result

    @test_api_version.larger_than("1.12.0")
    def test_supported_activities_large_scale(self):
        result = torch.profiler.supported_activities
        return result

