
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.supports_complex)
class TorchSupportsComplexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_supports_complex_correctness(self):
        result = torch.distributed.supports_complex()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_supports_complex_large_scale(self):
        result = torch.distributed.supports_complex()
        return result



