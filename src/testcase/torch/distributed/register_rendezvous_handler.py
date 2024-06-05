
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.register_rendezvous_handler)
class TorchRegisterRendezvousHandlerTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_register_rendezvous_handler_correctness(self):
        def rendezvous_handler(key):
            return key
        result = torch.distributed.register_rendezvous_handler(rendezvous_handler)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_register_rendezvous_handler_large_scale(self):
        def rendezvous_handler(key):
            return key
        result = torch.distributed.register_rendezvous_handler(rendezvous_handler)
        return result

