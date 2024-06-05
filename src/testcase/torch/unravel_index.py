
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.unravel_index)
class TorchUnravelIndexTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_unravel_index_correctness(self):
        indices = torch.randint(0, random.randint(1, 10) * random.randint(1, 10), (random.randint(1, 10),))
        shape = (random.randint(1, 10), random.randint(1, 10))
        result = torch.unravel_index(indices, shape)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_unravel_index_large_scale(self):
        indices = torch.randint(0, random.randint(1000, 10000) * random.randint(1000, 10000), (random.randint(1000, 10000),))
        shape = (random.randint(1000, 10000), random.randint(1000, 10000))
        result = torch.unravel_index(indices, shape)
        return result

