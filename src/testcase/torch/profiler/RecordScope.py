
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.RecordScope)
class TorchRecordScopeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_RecordScope_pybind11_type(self):
        result = torch.profiler.RecordScope.pybind11_type
        return result

    @test_api_version.larger_than("1.12.0")
    def test_RecordScope_name(self):
        result = torch.profiler.RecordScope.name
        return result

