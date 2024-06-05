
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.record_function)
class TorchRecordFunctionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.12.0")
    def test_record_function_type(self):
        result = torch.profiler.record_function.type
        return result

