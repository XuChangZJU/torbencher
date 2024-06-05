
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fx.Interpreter)
class TorchFxInterpreterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.11")
    def test_interpreter_correctness(self):
        # TODO: No concrete test case for Interpreter.type due to its abstract nature.
        return None

    @test_api_version.larger_than("1.11")
    def test_interpreter_large_scale(self):
        # TODO: No concrete test case for Interpreter.type due to its abstract nature.
        return None


