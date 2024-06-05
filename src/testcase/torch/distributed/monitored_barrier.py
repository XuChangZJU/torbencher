
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.monitored_barrier)
class TorchMonitoredBarrierTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_monitored_barrier_correctness(self):
        result = torch.distributed.monitored_barrier()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_monitored_barrier_large_scale(self):
        result = torch.distributed.monitored_barrier()
        return result

