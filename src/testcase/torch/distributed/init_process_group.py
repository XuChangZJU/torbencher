
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.init_process_group)
class TorchDistributedInitProcessGroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_init_process_group_0(self):
        a = 'nccl'
        result = torch.distributed.init_process_group(a, backend='nccl', world_size=4, rank=0)
        return result


