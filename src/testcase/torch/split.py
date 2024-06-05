
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.split)
class TorchSplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_split_correctness(self):
        tensor = torch.randn(random.randint(1, 10), random.randint(1, 10))
        split_size = random.randint(1, 10)
        result = torch.split(tensor, split_size)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_split_large_scale(self):
        tensor = torch.randn(random.randint(1000, 10000), random.randint(1000, 10000))
        split_size = random.randint(1000, 10000)
        result = torch.split(tensor, split_size)
        return result

