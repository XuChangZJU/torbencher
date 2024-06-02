
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.get_world_size)
class TorchDistributedGetWorldSizeTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_world_size_0(self, input=None):
        if input is not None:
            result = torch.distributed.get_world_size()
            return [result, input]
        result = torch.distributed.get_world_size()
        return [result, None]


