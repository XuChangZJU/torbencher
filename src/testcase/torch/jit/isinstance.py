
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.isinstance)
class TorchJitIsinstanceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_isinstance_correctness(self):
        class TestClass:
            pass
        result = torch.jit.isinstance(TestClass(), TestClass)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_isinstance_large_scale(self):
        class TestClass:
            pass
        result = torch.jit.isinstance(TestClass(), TestClass)
        return result

