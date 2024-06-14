import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.is_inference_mode_enabled)
class TorchIsinferencemodeenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_inference_mode_enabled_correctness(self):
        # No input parameters for torch.is_inference_mode_enabled()
        result_before = torch.is_inference_mode_enabled()
        with torch.inference_mode():
            result_inside = torch.is_inference_mode_enabled()
        result_after = torch.is_inference_mode_enabled()
        return result_before, result_inside, result_after
    
    
    
    
    
    
    