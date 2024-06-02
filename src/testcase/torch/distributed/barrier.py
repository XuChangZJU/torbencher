
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.barrier)
class TorchDistributedBarrierTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_barrier_0(self, input=None):
        if input is not None:
            result = torch.distributed.barrier()
            return [result, input]
        result = torch.distributed.barrier()
        return [result, None]


