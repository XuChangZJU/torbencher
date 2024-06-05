
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.RecursiveScriptClass)
class TorchJitRecursiveScriptClassTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_recursive_script_class_correctness(self):
        class TestClass(torch.jit.RecursiveScriptClass):
            def __init__(self):
                super().__init__()

        result = TestClass().type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_recursive_script_class_large_scale(self):
        class TestClass(torch.jit.RecursiveScriptClass):
            def __init__(self):
                super().__init__()

        result = TestClass().type
        return result

