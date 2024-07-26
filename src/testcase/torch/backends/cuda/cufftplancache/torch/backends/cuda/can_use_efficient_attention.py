import torch
import random


from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.can_use_efficient_attention)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaCanuseefficientattentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_can_use_efficient_attention(self):
        # This function checks if efficient attention can be used on the current device.
        result = torch.backends.cuda.can_use_efficient_attention(torch.backends.cuda.EfficientAttentionParams())
        return result
    
    
    
    