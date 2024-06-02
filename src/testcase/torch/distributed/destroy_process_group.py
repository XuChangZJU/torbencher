
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.destroy_process_group)
class TorchDistributedDestroyProcessGroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_destroy_process_group_0(self, input=None):
        if input is not None:
            result = torch.distributed.destroy_process_group()
            return [result, input]
        result = torch.distributed.destroy_process_group()
        return [result, None]


