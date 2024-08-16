import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.as_strided)
class TorchAsUstridedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_as_strided_correctness(self):
        x = torch.randn(3, 3)
        return torch.as_strided(x, (2, 2), (1, 2)), torch.as_strided(x, (2, 2), (1, 2), 1)
