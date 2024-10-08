import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.backends.cuda.cufft_plan_cache.torch.backends.cuda.can_use_efficient_attention)
class TorchBackendsCudaCufftplancacheTorchBackendsCudaCanUuseUefficientUattentionTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_can_use_efficient_attention(self):
        # This function checks if efficient attention can be used on the current device.
        result = torch.backends.cuda.can_use_efficient_attention(torch.backends.cuda.EfficientAttentionParams())
        return result
