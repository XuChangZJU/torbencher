
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.amp.custom_bwd)
class CustomBwdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.6.0")
    def test_custom_bwd(self):
        def custom_bwd(ctx, grad_output):
            return grad_output * 2
        result = torch.cuda.amp.custom_bwd(custom_bwd)
        return result

    @test_api_version.larger_than("1.6.0")
    def test_custom_bwd_large_scale(self):
        def custom_bwd(ctx, grad_output):
            return grad_output * 2
        result = torch.cuda.amp.custom_bwd(custom_bwd)
        return result

