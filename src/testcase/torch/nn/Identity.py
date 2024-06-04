
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.nn.Identity)
class TorchNNIdentityTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_identity(self):
        a = torch.randn(10)
        identity = torch.nn.Identity()
        result = identity(a)
        return result

