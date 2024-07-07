import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.onednn_fusion_enabled)
class TorchJitOnednnfusionenabledTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_onednn_fusion_enabled_correctness(self):
        # No input parameters for torch.jit.onednn_fusion_enabled
        result = torch.jit.onednn_fusion_enabled()
        return result
    
    
    
    