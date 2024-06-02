
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.distributed.get_backend)
class TorchDistributedGetBackendTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_get_backend_0(self, input=None):
        if input is not None:
            result = torch.distributed.get_backend()
            return [result, input]
        result = torch.distributed.get_backend()
        return [result, None]


