
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.all_gather_coalesced)
class TorchAllGatherCoalescedTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_all_gather_coalesced_correctness(self):
        dim = random.randint(1, 10)
        tensor_list = [torch.randn(dim) for _ in range(torch.distributed.get_world_size())]
        result = torch.distributed.all_gather_coalesced(tensor_list)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_all_gather_coalesced_large_scale(self):
        dim = random.randint(1000, 10000)
        tensor_list = [torch.randn(dim) for _ in range(torch.distributed.get_world_size())]
        result = torch.distributed.all_gather_coalesced(tensor_list)
        return result

