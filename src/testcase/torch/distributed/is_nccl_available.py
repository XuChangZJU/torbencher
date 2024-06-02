
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.is_nccl_available)
class TorchDistributedIsNcclAvailableTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_is_nccl_available_0(self, input=None):
        if input is not None:
            result = torch.distributed.is_nccl_available()
            return [result, input]
        result = torch.distributed.is_nccl_available()
        return [result, None]


