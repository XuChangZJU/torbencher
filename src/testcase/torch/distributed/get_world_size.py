
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.get_world_size)
class TorchGetWorldSizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_world_size_correctness(self):
        result = torch.distributed.get_world_size()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_world_size_large_scale(self):
        result = torch.distributed.get_world_size()
        return result

