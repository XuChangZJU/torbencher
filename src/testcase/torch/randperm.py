import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.randperm)
class TorchRandpermTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_randperm_correctness(self):
        # n should be larger than 0
        n = random.randint(1, 100)
        result = torch.randperm(n)
        return result
