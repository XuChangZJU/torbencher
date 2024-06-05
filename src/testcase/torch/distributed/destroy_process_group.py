
import torch
import random

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.destroy_process_group)
class TorchDestroyProcessGroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_destroy_process_group_correctness(self):
        result = torch.distributed.destroy_process_group()
        return result

    @test_api_version.larger_than("1.1.3")
    def test_destroy_process_group_large_scale(self):
        result = torch.distributed.destroy_process_group()
        return result

