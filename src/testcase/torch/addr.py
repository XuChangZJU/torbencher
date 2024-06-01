import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.addr)
class TorchAddrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_addr_4d(self, input=None):
        if input is not None:
            result = torch.addr(input[0], input[1], input[2])
            return [result, input]
        a = torch.randn(4, 5)
        b = torch.randn(4)
        c = torch.randn(5)
        result = torch.addr(a, b, c)
        return [result, [a, b, c]]
