
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.classproperty)
class TorchClasspropertyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_classproperty_correctness(self):
        class TestClass:
            @torch.classproperty
            def test_prop(cls):
                return random.randint(1, 10)
        result = TestClass.test_prop
        return result

    @test_api_version.larger_than("1.1.3")
    def test_classproperty_large_scale(self):
        class TestClass:
            @torch.classproperty
            def test_prop(cls):
                return random.randint(1, 10)
        result = TestClass.test_prop
        return result

