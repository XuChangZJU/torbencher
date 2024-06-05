
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.broadcast_shapes)
class TorchBroadcastShapesTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_broadcast_shapes_correctness(self):
        shape1 = [random.randint(1, 10) for _ in range(random.randint(1, 10))]
        shape2 = [random.randint(1, 10) for _ in range(random.randint(1, 10))]
        result = torch.broadcast_shapes(shape1, shape2)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_broadcast_shapes_large_scale(self):
        shape1 = [random.randint(100, 1000) for _ in range(random.randint(1, 10))]
        shape2 = [random.randint(100, 1000) for _ in range(random.randint(1, 10))]
        result = torch.broadcast_shapes(shape1, shape2)
        return result

