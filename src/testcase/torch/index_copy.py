
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.index_copy)
class TorchIndexCopyTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_index_copy_correctness(self):
        input = torch.randn(random.randint(1, 10), random.randint(1, 10))
        dim = random.randint(0, 1)
        index = torch.randint(0, random.randint(1, 10), (random.randint(1, 10),))
        source = torch.randn(random.randint(1, 10), random.randint(1, 10))
        result = torch.index_copy(input, dim, index, source)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_index_copy_large_scale(self):
        input = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        dim = random.randint(0, 1)
        index = torch.randint(0, random.randint(1000, 10000), (random.randint(1000, 10000),))
        source = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        result = torch.index_copy(input, dim, index, source)
        return result

