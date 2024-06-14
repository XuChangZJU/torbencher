import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.getdeterministicdebugmode)
class TorchGetdeterministicdebugmodeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_deterministic_debug_mode_correctness(self):
    # No random parameters needed for this test case
    torch.set_deterministic_debug_mode(True)
    result_true = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode(False)
    result_false = torch.get_deterministic_debug_mode()
    return result_true, result_false # Returning both results to show the effect of setting the debug mode
