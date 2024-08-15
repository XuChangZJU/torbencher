import random

import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.broadcast_to)
class TorchBroadcastUtoTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("2.0.0")
    def test_broadcast_to_correctness(self):
        x = torch.tensor([1, 2, 3])
        return torch.broadcast_to(x, (3, 3))