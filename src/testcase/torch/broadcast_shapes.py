import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api


@test_api(torch.broadcast_shapes)
class TorchBroadcastUshapesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_shapes_correctness(self):
        return torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
