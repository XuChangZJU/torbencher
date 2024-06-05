
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.TCPStore)
class TorchTCPStoreTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_tcp_store_correctness(self):
        host = 'localhost'
        port = random.randint(10000, 65535)
        result = torch.distributed.TCPStore.pybind11_type(host, port)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_tcp_store_large_scale(self):
        host = 'localhost'
        port = random.randint(10000, 65535)
        result = torch.distributed.TCPStore.pybind11_type(host, port)
        return result

