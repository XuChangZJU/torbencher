import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.vsplit)
class TorchVsplitTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_vsplit_correctness(self):
        t = torch.arange(16.0).reshape(4,4)
        return t, torch.vsplit(t, 2), torch.vsplit(t, [3, 6])
