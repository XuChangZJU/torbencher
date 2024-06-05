
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.narrow_copy)
class TorchNarrowCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_narrow_copy_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        dim = random.randint(0, 2)
        start = random.randint(0, random.randint(1, 10))
        length = random.randint(1, 10)
        result = torch.narrow_copy(input, dim, start, length)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_narrow_copy_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000), random.randint(1000, 10000))
        dim = random.randint(0, 2)
        start = random.randint(0, random.randint(1000, 10000))
        length = random.randint(1000, 10000)
        result = torch.narrow_copy(input, dim, start, length)
        return result

