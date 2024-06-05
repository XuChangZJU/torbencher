
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.new_subgroups)
class TorchNewSubgroupsTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_new_subgroups_correctness(self):
        num_subgroups = random.randint(1, torch.distributed.get_world_size() - 1)
        result = torch.distributed.new_subgroups(num_subgroups)
        return result

    @test_api_version.larger_than("1.1.3")
    def test_new_subgroups_large_scale(self):
        num_subgroups = random.randint(10, torch.distributed.get_world_size() - 1)
        result = torch.distributed.new_subgroups(num_subgroups)
        return result

