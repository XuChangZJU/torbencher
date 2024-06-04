
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.all_reduce)
class TorchDistributedAllReduceTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_reduce_0(self):
        a = torch.tensor([1, 2, 3, 4])
        b = torch.distributed.reduce_op.SUM
        result = torch.distributed.all_reduce(a, op=b)
        return result


