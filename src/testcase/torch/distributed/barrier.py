
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.barrier)
class TorchDistributedBarrierTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_barrier_0(self):
        
        result = torch.distributed.barrier()
        return result


