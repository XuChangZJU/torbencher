
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.reduce)
class TorchDistributedReduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reduce_0(self, input=None):
        if input is not None:
            result = torch.distributed.reduce(input[0], dst=input[1], op=input[2])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4])
        b = 1
        c = torch.distributed.reduce_op.SUM
        result = torch.distributed.reduce(a, dst=b, op=c)
        return [result, [a, b, c]]


