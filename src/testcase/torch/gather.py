import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.gather)
class TorchGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_gather_correctness(self):
        t = torch.tensor([[1, 2], [3, 4]])
        return torch.gather(t, 1, torch.tensor([[0, 0], [1, 0]]))
