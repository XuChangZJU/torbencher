
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.gather)
class TorchDistributedGatherTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_gather_0(self):
        a = torch.tensor([1, 2, 3, 4])
        b = [torch.tensor([1, 2, 3, 4]) for _ in range(4)]
        c = 0
        d = torch.distributed.group.WORLD
        result = torch.distributed.gather(a, gather_list=b, dst=c, group=d)
        return result


