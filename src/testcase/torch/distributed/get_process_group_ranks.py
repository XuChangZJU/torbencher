
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.get_process_group_ranks)
class TorchGetProcessGroupRanksTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_process_group_ranks_correctness(self):
        result = torch.distributed.get_process_group_ranks()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_get_process_group_ranks_large_scale(self):
        result = torch.distributed.get_process_group_ranks()
        return result

