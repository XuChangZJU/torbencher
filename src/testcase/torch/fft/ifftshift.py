
import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.fft.ifftshift)
class TorchIfftshiftTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.7.0")
    def test_ifftshift_4d(self, input=None):
        if input is not None:
            result = torch.fft.ifftshift(input[0], input[1])
            return [result, input]
        a = torch.randn(4, 3, 8, 8)
        b = None
        result = torch.fft.ifftshift(a, dim=b)
        return [result, [a, b]]






