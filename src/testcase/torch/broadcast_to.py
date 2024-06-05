
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.broadcast_to)
class TorchBroadcastToTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_to_correctness(self):
        dim = random.randint(1, 10)
        tensor = torch.randn(dim)
        shape = [random.randint(1, 10) for _ in range(random.randint(1, 10))]
        result = torch.broadcast_to(tensor, shape)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_broadcast_to_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor = torch.randn(dim)
        shape = [random.randint(100, 1000) for _ in range(random.randint(1, 10))]
        result = torch.broadcast_to(tensor, shape)
        return result

