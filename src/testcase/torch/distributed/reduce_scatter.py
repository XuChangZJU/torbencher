
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.reduce_scatter)
class TorchDistributedReduceScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_reduce_scatter_0(self, input=None):
        if input is not None:
            result = torch.distributed.reduce_scatter(input[0], input[1], op=input[2])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4])
        b = [torch.tensor([1, 2, 3, 4]) for _ in range(4)]
        c = torch.distributed.ReduceOp.SUM
        result = torch.distributed.reduce_scatter(a, b, op=c)
        return [result, [a, b, c]]
