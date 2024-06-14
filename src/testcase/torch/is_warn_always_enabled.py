import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_warn_always_enabled)
class TorchIswarnalwaysenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_warn_always_enabled_correctness(self):
        # No random parameters needed for this test case
        # Check the initial state
        initial_state = torch.is_warn_always_enabled()
    
        # Toggle the state and check
        torch.set_warn_always(not initial_state)
        toggled_state = torch.is_warn_always_enabled()
    
        # Assert that the state has been toggled
        assert toggled_state != initial_state
    
        # Return the final state
        return torch.is_warn_always_enabled()
    
    
    
    
    
    
    