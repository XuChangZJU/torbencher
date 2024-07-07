import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.profiler.itt.mark)
class TorchProfilerIttMarkTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_mark_correctness(self):
        msg = "test message" # message (str): ASCII message to associate with the event.
        result = torch.profiler.itt.mark(msg)
        return result
    
    
    
    