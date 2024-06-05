
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.jit.Attribute)
class TorchJitAttributeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_attribute_correctness(self):
        attr = torch.jit.Attribute(torch.randn(random.randint(1, 10)))
        result = attr.type
        return result

    @test_api_version.larger_than("1.1.3")
    def test_attribute_large_scale(self):
        attr = torch.jit.Attribute(torch.randn(random.randint(1000, 10000)))
        result = attr.type
        return result

