import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.backends.mha.set_fastpath_enabled)
class TorchBackendsMhaSetfastpathenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_set_fastpath_enabled_correctness(self):
        # Randomly enable or disable the fastpath
        enable_fastpath = random.choice([True, False])
        
        # Set the fastpath enabled/disabled
        torch._C._jit_set_profiling_executor(enable_fastpath)
        
        # Verify the setting by checking the current state
        current_state = torch._C._jit_get_profiling_executor()
        
        # Return the current state to verify correctness
        return current_state
    
    
    
    