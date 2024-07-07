import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.enable_onednn_fusion)
class TorchJitEnableonednnfusionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_enable_onednn_fusion_correctness(self):
        # Randomly enable or disable onednn JIT fusion
        enabled_status = random.choice([True, False]) 
        result = torch.jit.enable_onednn_fusion(enabled_status)
        return result
    
    
    
    