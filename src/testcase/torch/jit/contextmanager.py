
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.contextmanager)
class TorchJitContextmanagerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_contextmanager_correctness(self):
        @torch.jit.contextmanager
        def test_context():
            yield 1
        result = test_context()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_contextmanager_large_scale(self):
        @torch.jit.contextmanager
        def test_context():
            yield 1
        result = test_context()
        return result

