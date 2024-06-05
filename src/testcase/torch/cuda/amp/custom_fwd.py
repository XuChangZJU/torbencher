
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.cuda.amp.custom_fwd)
class CustomFwdTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.6.0")
    def test_custom_fwd(self):
        def custom_fwd(ctx, input, other):
            return input + other
        result = torch.cuda.amp.custom_fwd(custom_fwd)
        return result

    @test_api_version.larger_than("1.6.0")
    def test_custom_fwd_large_scale(self):
        def custom_fwd(ctx, input, other):
            return input + other
        result = torch.cuda.amp.custom_fwd(custom_fwd)
        return result



