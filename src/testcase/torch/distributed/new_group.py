
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.new_group)
class TorchDistributedNewGroupTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_new_group_0(self, input=None):
        if input is not None:
            result = torch.distributed.new_group(input[0])
            return [result, input]
        a = [0, 1, 2, 3]
        result = torch.distributed.new_group(a)
        return [result, [a]]




