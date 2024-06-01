import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.qr)
class TorchQrTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_qr_4d(self, input=None):
        if input is not None:
            result = torch.qr(input[0])
            return [result, input]
        a = torch.randn(4, 4)
        result = torch.qr(a)
        return [result, [a]]

