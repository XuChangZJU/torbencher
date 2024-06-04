
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.all_to_all)
class TorchDistributedAllToAllTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_to_all_0(self):
        
        a = torch.tensor([1, 2, 3, 4])
        b = torch.tensor([4, 3, 2, 1])
        c = torch.distributed.group.WORLD
        result = torch.distributed.all_to_all(a, b, group=c)
        return result


