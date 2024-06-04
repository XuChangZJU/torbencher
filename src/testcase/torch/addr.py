
import torch

from src.testcase.TorBencherTestCaseBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addr)
class TorchAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr(self):
        vec1 = torch.arange(1., 4.)
        vec2 = torch.arange(1., 3.)
        M = torch.zeros(3, 2)
        result = torch.addr(M, vec1, vec2, beta=10, alpha=0.5)
        return result

