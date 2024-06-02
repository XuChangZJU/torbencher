
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.is_initialized)
class TorchDistributedIsInitializedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_initialized_0(self, input=None):
        if input is not None:
            result = torch.distributed.is_initialized()
            return [result, input]
        result = torch.distributed.is_initialized()
        return [result, None]


