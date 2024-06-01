import torch

from src.testcase.TorBencherBase import TorBencherTestCaseBase
from src.util import test_api_version
from src.util.decorator import test_api

@test_api(torch.erfinv)
class TorchErfinvTestCase(TorBencherTestCaseBase):
    @test_api_version.larger_than("1.1.3")
    def test_erfinv_4d(self, input=None):
        if input is not None:
            result = torch.erfinv(input[0])
            return [result, input]
        a = torch.randn(4).clamp(-0.999, 0.999)
        result = torch.erfinv(a)
        return [result, [a]]

