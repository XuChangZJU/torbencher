
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.scatter)
class TorchDistributedScatterTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_scatter_0(self, input=None):
        if input is not None:
            result = torch.distributed.scatter(input[0], src=input[1], scatter_list=input[2], group=input[3])
            return [result, input]
        a = torch.tensor([1, 2, 3, 4])
        b = 0
        c = [torch.tensor([1, 2, 3, 4]) for _ in range(4)]
        d = torch.distributed.group.WORLD
        result = torch.distributed.scatter(a, src=b, scatter_list=c, group=d)
        return [result, [a, b, c, d]]


