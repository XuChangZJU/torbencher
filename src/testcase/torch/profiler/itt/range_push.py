import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.itt.range_push)
class TorchProfilerIttRangepushTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_range_push_correctness(self):
        # No random parameters for torch.profiler.itt.range_push
        msg = "test_message" # type: str
        result = torch.profiler.itt.range_push(msg)
        return result
    
    
    
    