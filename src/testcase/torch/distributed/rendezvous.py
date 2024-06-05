
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.rendezvous)
class TorchRendezvousTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_rendezvous_correctness(self):
        key = 'test_rendezvous'
        result = torch.distributed.rendezvous(key)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_rendezvous_large_scale(self):
        key = 'test_rendezvous'
        result = torch.distributed.rendezvous(key)
        return result

