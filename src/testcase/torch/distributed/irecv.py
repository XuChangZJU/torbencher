
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.irecv)
class TorchDistributedIrecvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_irecv_0(self, input=None):
        if input is not None:
            result = torch.distributed.irecv(input[0], src=input[1])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4])
        b = 0
        result = torch.distributed.irecv(a, src=b)
        return [result, [a, b]]

