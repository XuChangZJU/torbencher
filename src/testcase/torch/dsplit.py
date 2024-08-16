import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.dsplit)
class TorchDsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_dsplit_correctness(self):
        t = torch.arange(16.0).reshape(2, 2, 4)
        return torch.dsplit(t, 2), torch.dsplit(t, [3, 6])

