
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.recv)
class TorchDistributedRecvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_recv_0(self, input=None):
        if input is not None:
            result = torch.distributed.recv(input[0], src=input[1])
            return result
        a = torch.tensor([1, 2, 3, 4])
        b = 0
        result = torch.distributed.recv(a, src=b)
        return result


